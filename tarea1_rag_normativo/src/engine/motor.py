r"""Motor RAG normativo — módulo único de lógica de negocio.

`responder(pregunta)` es el ÚNICO punto de entrada que debe usar cualquier
interfaz (Streamlit, notebook, CLI, lo que sea). Este módulo NUNCA importa
streamlit ni ninguna librería de UI — verificable con (busca imports reales,
no menciones en comentarios como esta):

    grep -RInE "^\s*(import|from)\s+(streamlit|tkinter|flask)" src/engine/motor.py   (sin resultados)

Contrato del resultado que devuelve `responder()` (todas las llaves
siempre presentes, para que la interfaz nunca tenga que inferir nada del
texto de la respuesta):
    pregunta, fuentes, mejor_similitud, umbral_similitud,
    abstuvo (bool estructurado), respuesta, tokens_entrada, tokens_salida,
    costo_usd, franja_horaria, latencia_ms

Si la llamada al proveedor de generación falla, `responder()` LANZA
`ErrorDeAPI` en vez de devolver un dict disfrazado de respuesta normal —
la interfaz debe capturarla y mostrarla como error.

El proveedor de generación se elige en config.yaml (`generacion.proveedor`:
"gemini" o "deepseek"). Cambiar de proveedor es un cambio de configuración,
no de código.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from ingest.embeddings import crear_embedder
from ingest.vectorstore import abrir_coleccion

from .costos import RegistroLlamada, ahora_utc_iso, registrar_llamada
from .pricing import calcular_costo_deepseek, calcular_costo_gemini

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

_coleccion_cache = None


class ErrorDeAPI(Exception):
    """Falla del proveedor de generación (red, autenticación, cuota,
    respuesta inválida, etc.). Nunca se convierte en una respuesta normal."""

    def __init__(self, mensaje: str, proveedor: str, status_code: int | None = None):
        super().__init__(mensaje)
        self.proveedor = proveedor
        self.status_code = status_code


def cargar_config(ruta: str | Path = RAIZ_PROYECTO / "config.yaml") -> dict:
    with open(ruta, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _obtener_coleccion(config: dict):
    global _coleccion_cache
    if _coleccion_cache is None:
        proveedor_embeddings = config["embeddings"]["proveedor"]
        _coleccion_cache = abrir_coleccion(
            RAIZ_PROYECTO / config["rutas"]["indice_vectorial"],
            embedding_function=crear_embedder(proveedor_embeddings, modo="passage"),
        )
    return _coleccion_cache


def _recuperar_fragmentos(pregunta: str, config: dict) -> list[dict]:
    coleccion = _obtener_coleccion(config)
    # IMPORTANTE: cambiar `embeddings.proveedor` en config.yaml solo tiene
    # efecto real si `rutas.indice_vectorial` apunta a un índice construido
    # CON ESE MISMO proveedor (los vectores de un índice no son
    # intercambiables entre modelos — distinta dimensión, distinta escala
    # de similitud; ver Fase 4). No basta con cambiar este campo solo.
    embedder_query = crear_embedder(config["embeddings"]["proveedor"], modo="query")
    vector = embedder_query([pregunta])[0]
    resultados = coleccion.query(query_embeddings=[vector], n_results=config["recuperacion"]["k"])

    fragmentos = []
    for doc, meta, dist in zip(
        resultados["documents"][0], resultados["metadatas"][0], resultados["distances"][0]
    ):
        fragmentos.append({**meta, "texto": doc, "similitud": 1 - dist})
    return fragmentos


def _construir_mensajes(pregunta: str, fragmentos: list[dict], config: dict) -> list[dict]:
    bloques = [
        f"[Fragmento {i}] Documento: {f['titulo']} ({f['tipo']}, versión {f['version']}), "
        f"página {f['pagina']}.\n{f['texto']}"
        for i, f in enumerate(fragmentos, start=1)
    ]
    prompt_usuario = f"CONTEXTO:\n{chr(10).join(bloques)}\n\nPREGUNTA:\n{pregunta}\n\nRESPUESTA:"
    return [
        {"role": "system", "content": config["mensajes"]["system_prompt"]},
        {"role": "user", "content": prompt_usuario},
    ]


def _llamar_deepseek(mensajes: list[dict], config: dict) -> dict:
    """Devuelve {texto, tokens_entrada_cache_hit, tokens_entrada_cache_miss, tokens_salida}."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ErrorDeAPI("Falta DEEPSEEK_API_KEY en el entorno (.env)", proveedor="deepseek")

    url = config["generacion"]["api_base"].rstrip("/") + "/chat/completions"
    payload = {
        "model": config["generacion"]["modelo"],
        "messages": mensajes,
        "temperature": config["generacion"]["temperatura"],
        "max_tokens": config["generacion"]["max_tokens_salida"],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        respuesta = requests.post(url, json=payload, headers=headers, timeout=30)
    except requests.RequestException as err:
        raise ErrorDeAPI(f"Error de red llamando a DeepSeek: {err}", proveedor="deepseek") from err

    if respuesta.status_code != 200:
        raise ErrorDeAPI(
            f"DeepSeek respondió {respuesta.status_code}: {respuesta.text[:300]}",
            proveedor="deepseek",
            status_code=respuesta.status_code,
        )

    cruda = respuesta.json()
    uso = cruda.get("usage", {})
    tokens_hit = uso.get("prompt_cache_hit_tokens", 0)
    tokens_miss = uso.get("prompt_cache_miss_tokens", max(uso.get("prompt_tokens", 0) - tokens_hit, 0))
    return {
        "texto": cruda["choices"][0]["message"]["content"],
        "tokens_entrada_cache_hit": tokens_hit,
        "tokens_entrada_cache_miss": tokens_miss,
        "tokens_salida": uso.get("completion_tokens", 0),
    }


def _llamar_gemini(mensajes: list[dict], config: dict) -> dict:
    """Devuelve {texto, tokens_entrada_cache_hit, tokens_entrada_cache_miss, tokens_salida}.
    Gemini no distingue cache hit/miss como DeepSeek: todo el input va como
    'cache_miss' para que la tabla de costos lo trate igual (en la capa
    gratuita el costo es $0 de todas formas, ver pricing.py)."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ErrorDeAPI("Falta GEMINI_API_KEY en el entorno (.env)", proveedor="gemini")

    try:
        from google import genai
        from google.genai import types
    except ImportError as err:
        raise ErrorDeAPI(f"Falta instalar google-genai: {err}", proveedor="gemini") from err

    system_prompt = next(m["content"] for m in mensajes if m["role"] == "system")
    prompt_usuario = next(m["content"] for m in mensajes if m["role"] == "user")

    try:
        cliente = genai.Client(api_key=api_key)
        respuesta = cliente.models.generate_content(
            model=config["generacion"]["modelo"],
            contents=prompt_usuario,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=config["generacion"]["temperatura"],
                max_output_tokens=config["generacion"]["max_tokens_salida"],
            ),
        )
    except Exception as err:  # cualquier falla del SDK/red/cuota de Gemini
        raise ErrorDeAPI(f"Gemini falló: {err}", proveedor="gemini") from err

    texto = respuesta.text
    if texto is None:
        raise ErrorDeAPI(
            f"Gemini no devolvió texto (posible bloqueo de seguridad): {respuesta}",
            proveedor="gemini",
        )

    uso = getattr(respuesta, "usage_metadata", None)
    tokens_entrada = getattr(uso, "prompt_token_count", 0) or 0
    tokens_salida = getattr(uso, "candidates_token_count", 0) or 0

    return {
        "texto": texto,
        "tokens_entrada_cache_hit": 0,
        "tokens_entrada_cache_miss": tokens_entrada,
        "tokens_salida": tokens_salida,
    }


def _fuentes_para_resultado(fragmentos: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": f["doc_id"],
            "titulo": f["titulo"],
            "pagina": f["pagina"],
            "version": f["version"],
            "similitud": round(f["similitud"], 4),
        }
        for f in fragmentos
    ]


def _calcular_costo(config: dict, resultado_llm: dict, momento_utc: datetime) -> dict:
    proveedor = config["generacion"]["proveedor"]
    if proveedor == "gemini":
        return calcular_costo_gemini(
            modelo=config["generacion"]["modelo"],
            tokens_entrada=resultado_llm["tokens_entrada_cache_miss"],
            tokens_salida=resultado_llm["tokens_salida"],
            nivel=config["generacion"].get("nivel", "pagado"),
        )
    if proveedor == "deepseek":
        return calcular_costo_deepseek(
            modelo=config["generacion"]["modelo"],
            tokens_entrada_cache_hit=resultado_llm["tokens_entrada_cache_hit"],
            tokens_entrada_cache_miss=resultado_llm["tokens_entrada_cache_miss"],
            tokens_salida=resultado_llm["tokens_salida"],
            momento_utc=momento_utc,
        )
    raise ValueError(f"Proveedor de generación desconocido: {proveedor!r}")


def responder(pregunta: str, config: dict | None = None) -> dict:
    if config is None:
        config = cargar_config()

    fragmentos = _recuperar_fragmentos(pregunta, config)
    mejor_similitud = fragmentos[0]["similitud"] if fragmentos else 0.0
    umbral = config["recuperacion"]["umbral_similitud"]

    base = {
        "pregunta": pregunta,
        "fuentes": _fuentes_para_resultado(fragmentos),
        "mejor_similitud": round(mejor_similitud, 4),
        "umbral_similitud": umbral,
    }

    if mejor_similitud < umbral:
        # Decisión ANTES de llamar al LLM: no se gasta ni un token de
        # generación en una pregunta fuera del corpus.
        return {
            **base,
            "abstuvo": True,
            "respuesta": config["mensajes"]["abstencion_fuera_de_indice"],
            "tokens_entrada": 0,
            "tokens_salida": 0,
            "costo_usd": 0.0,
            "franja_horaria": None,
            "latencia_ms": 0.0,
        }

    mensajes = _construir_mensajes(pregunta, fragmentos, config)
    proveedor = config["generacion"]["proveedor"]
    llamador = {"deepseek": _llamar_deepseek, "gemini": _llamar_gemini}[proveedor]

    momento_utc = datetime.now(timezone.utc)
    inicio = time.monotonic()

    try:
        resultado_llm = llamador(mensajes, config)
    except ErrorDeAPI as err:
        registrar_llamada(RegistroLlamada(
            timestamp_utc=ahora_utc_iso(),
            modelo=config["generacion"]["modelo"],
            franja_horaria="no_aplica",
            tokens_entrada_cache_hit=0, tokens_entrada_cache_miss=0, tokens_salida=0,
            latencia_ms=round((time.monotonic() - inicio) * 1000, 1),
            costo_usd=0.0, exito=False, detalle_error=str(err),
        ))
        raise  # nunca se convierte en una respuesta normal

    latencia_ms = (time.monotonic() - inicio) * 1000
    info_costo = _calcular_costo(config, resultado_llm, momento_utc)

    registrar_llamada(RegistroLlamada(
        timestamp_utc=ahora_utc_iso(),
        modelo=config["generacion"]["modelo"],
        franja_horaria=info_costo["franja_horaria"],
        tokens_entrada_cache_hit=resultado_llm["tokens_entrada_cache_hit"],
        tokens_entrada_cache_miss=resultado_llm["tokens_entrada_cache_miss"],
        tokens_salida=resultado_llm["tokens_salida"],
        latencia_ms=round(latencia_ms, 1),
        costo_usd=info_costo["costo_usd"], exito=True,
        fuente_precios=info_costo["fuente_precios"],
        fecha_verificacion_precios=info_costo["fecha_verificacion_precios"],
    ))

    return {
        **base,
        "abstuvo": False,
        "respuesta": resultado_llm["texto"],
        "tokens_entrada": resultado_llm["tokens_entrada_cache_hit"] + resultado_llm["tokens_entrada_cache_miss"],
        "tokens_salida": resultado_llm["tokens_salida"],
        "costo_usd": info_costo["costo_usd"],
        "franja_horaria": info_costo["franja_horaria"],
        "latencia_ms": round(latencia_ms, 1),
    }

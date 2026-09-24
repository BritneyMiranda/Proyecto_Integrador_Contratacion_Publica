r"""Motor RAG híbrido de la Tarea 2 — módulo único de lógica de negocio.

`responder(pregunta)` es el único punto de entrada. Reutiliza LITERALMENTE
piezas del motor de la Tarea 1 (no las reimplementa):

- `ErrorDeAPI`, `_llamar_gemini`, `_llamar_deepseek` — de
  `tarea1_rag_normativo/src/engine/motor.py` (llamar al LLM y manejar sus
  errores es idéntico sea cual sea el corpus).
- `RegistroLlamada`, `registrar_llamada` — de `.../engine/costos.py`
  (mismo formato de log, apuntado al `logs/costos.csv` DE ESTA tarea, no el
  de la Tarea 1 — se pasa `ruta_log` explícito).
- `calcular_costo_gemini`, `calcular_costo_deepseek` — de `.../engine/pricing.py`.
- `E5Embedder`, `abrir_coleccion` — de `.../ingest/`.

Lo que SÍ es propio de esta tarea: `filtros.py` (separar condiciones
numéricas/territoriales de la búsqueda semántica) y el prompt de citación
por `ocid` en vez de documento+página.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAIZ_TAREA2 = Path(__file__).resolve().parents[1]
RAIZ_TAREA1 = RAIZ_TAREA2.parent / "tarea1_rag_normativo"
sys.path.insert(0, str(RAIZ_TAREA1 / "src"))
sys.path.insert(0, str(RAIZ_TAREA2 / "src"))

import yaml  # noqa: E402

from ingest.embeddings import E5Embedder  # noqa: E402
from ingest.vectorstore import abrir_coleccion  # noqa: E402

from engine.costos import RegistroLlamada, ahora_utc_iso, registrar_llamada  # noqa: E402
from engine.motor import ErrorDeAPI, _llamar_deepseek, _llamar_gemini  # noqa: E402
from engine.pricing import calcular_costo_deepseek, calcular_costo_gemini  # noqa: E402

from filtros import extraer_filtros  # noqa: E402

RUTA_LOG_COSTOS = RAIZ_TAREA2 / "logs" / "costos.csv"

_coleccion_cache = None


def cargar_config(ruta: str | Path = RAIZ_TAREA2 / "config.yaml") -> dict:
    with open(ruta, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _obtener_coleccion(config: dict):
    global _coleccion_cache
    if _coleccion_cache is None:
        _coleccion_cache = abrir_coleccion(
            RAIZ_TAREA2 / config["rutas"]["indice_vectorial"],
            embedding_function=E5Embedder(modo="passage"),
        )
    return _coleccion_cache


def _recuperar_procesos(pregunta: str, config: dict) -> tuple[list[dict], dict | None]:
    coleccion = _obtener_coleccion(config)
    where = extraer_filtros(pregunta, config["filtros"]["alias_categoria"])

    vector = E5Embedder(modo="query")([pregunta])[0]
    argumentos = {"query_embeddings": [vector], "n_results": config["recuperacion"]["k"]}
    if where:
        argumentos["where"] = where
    resultados = coleccion.query(**argumentos)

    procesos = []
    if resultados["ids"][0]:
        for doc, meta, dist in zip(
            resultados["documents"][0], resultados["metadatas"][0], resultados["distances"][0]
        ):
            procesos.append({**meta, "texto": doc, "similitud": 1 - dist})
    return procesos, where


def _construir_mensajes(pregunta: str, procesos: list[dict], config: dict) -> list[dict]:
    bloques = [
        f"[Proceso {i}] ocid: {p['ocid']} | departamento: {p['departamento']} | "
        f"categoría: {p['categoria']} | monto: {p['monto']:,.2f} {('PEN' if p['monto'] else '')} | "
        f"comprador: {p['comprador_nombre']}\n{p['texto']}"
        for i, p in enumerate(procesos, start=1)
    ]
    prompt_usuario = f"CONTEXTO:\n{chr(10).join(bloques)}\n\nPREGUNTA:\n{pregunta}\n\nRESPUESTA:"
    return [
        {"role": "system", "content": config["mensajes"]["system_prompt"]},
        {"role": "user", "content": prompt_usuario},
    ]


def _fuentes_para_resultado(procesos: list[dict]) -> list[dict]:
    return [
        {
            "ocid": p["ocid"],
            "titulo": p.get("titulo", ""),
            "departamento": p["departamento"],
            "categoria": p["categoria"],
            "monto": p["monto"],
            "comprador_nombre": p["comprador_nombre"],
            "similitud": round(p["similitud"], 4),
        }
        for p in procesos
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

    procesos, filtros_aplicados = _recuperar_procesos(pregunta, config)
    mejor_similitud = procesos[0]["similitud"] if procesos else 0.0
    umbral = config["recuperacion"]["umbral_similitud"]

    base = {
        "pregunta": pregunta,
        "filtros_aplicados": filtros_aplicados,
        "fuentes": _fuentes_para_resultado(procesos),
        "mejor_similitud": round(mejor_similitud, 4),
        "umbral_similitud": umbral,
    }

    if mejor_similitud < umbral:
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

    mensajes = _construir_mensajes(pregunta, procesos, config)
    proveedor = config["generacion"]["proveedor"]
    llamador = {"deepseek": _llamar_deepseek, "gemini": _llamar_gemini}[proveedor]

    momento_utc = datetime.now(timezone.utc)
    inicio = time.monotonic()

    try:
        resultado_llm = llamador(mensajes, config)
    except ErrorDeAPI:
        registrar_llamada(
            RegistroLlamada(
                timestamp_utc=ahora_utc_iso(),
                modelo=config["generacion"]["modelo"],
                franja_horaria="no_aplica",
                tokens_entrada_cache_hit=0, tokens_entrada_cache_miss=0, tokens_salida=0,
                latencia_ms=round((time.monotonic() - inicio) * 1000, 1),
                costo_usd=0.0, exito=False, detalle_error="ver excepción",
            ),
            ruta_log=RUTA_LOG_COSTOS,
        )
        raise

    latencia_ms = (time.monotonic() - inicio) * 1000
    info_costo = _calcular_costo(config, resultado_llm, momento_utc)

    registrar_llamada(
        RegistroLlamada(
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
        ),
        ruta_log=RUTA_LOG_COSTOS,
    )

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

"""Fase 4 — comparación obligatoria: mismo corpus, mismos fragmentos, dos
modelos de embeddings (local vs. API). Construye DOS índices ChromaDB
frescos con EXACTAMENTE los mismos fragmentos (misma segmentación
determinística de Fase 2: chunk_size=128, chunk_overlap=24), cada uno
embebido con un modelo distinto, y mide:

- Recall@1/@3/@5 (vía eval/evaluar.py, sin llamar al LLM de generación)
- Tiempo de indexación (segundos, pared de reloj)
- Costo real en USD (local: $0 siempre; API: tokens reales devueltos por
  la propia API × precio verificado, no una estimación)
- Latencia promedio de consulta (ms)
- Dimensión del vector (tomada directamente de la respuesta real, no
  supuesta)

Nota sobre el proveedor "API": el enunciado pide `text-embedding-3-small`
de OpenAI. La cuenta de OpenAI disponible no tenía saldo prepago (OpenAI no
tiene capa gratuita para su API, ni de prueba), así que la corrida real se
hizo con Gemini (`gemini-embedding-2`, capa gratuita real) para poder
reportar números medidos, no inventados. El código de OpenAI queda
completo y listo (`api_proveedor="openai"`) para correr en cuanto haya
saldo — ver README para la discusión completa de esta sustitución.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PROYECTO / "src"))
sys.path.insert(0, str(RAIZ_PROYECTO / "eval"))

from dotenv import load_dotenv

load_dotenv(RAIZ_PROYECTO / ".env")

from ingest.chunking import segmentar_paginas
from ingest.embeddings import (
    FECHA_VERIFICACION_PRECIO_GEMINI,
    FECHA_VERIFICACION_PRECIO_OPENAI,
    FUENTE_PRECIO_GEMINI,
    FUENTE_PRECIO_OPENAI,
    PRECIO_GEMINI_USD_POR_1M_TOKENS_PAGADO,
    PRECIO_OPENAI_USD_POR_1M_TOKENS,
    E5Embedder,
    GeminiEmbedder,
    OpenAIEmbedder,
)
from ingest.vectorstore import abrir_coleccion, indexar_fragmentos
from evaluar import evaluar

CHUNK_SIZE, CHUNK_OVERLAP = 128, 24
DIR_PROCESSED = RAIZ_PROYECTO / "data" / "processed"


def _cargar_paginas(doc_id: str) -> list[dict]:
    ruta = DIR_PROCESSED / f"{doc_id}.jsonl"
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines()]


def _fragmentos_del_corpus() -> list[dict]:
    """Misma llamada, mismos parámetros -> mismos fragmentos, mismos
    chunk_id, para los dos índices. Esto es lo que garantiza que la
    comparación sea justa (Fase 2: chunking determinístico)."""
    fragmentos = []
    for doc_id in ("ley_32069", "ds_001_2026_ef"):
        fragmentos += segmentar_paginas(_cargar_paginas(doc_id), CHUNK_SIZE, CHUNK_OVERLAP)
    return fragmentos


def _construir_indice(nombre: str, ruta_indice: Path, embedder_passage) -> dict:
    shutil.rmtree(ruta_indice, ignore_errors=True)
    fragmentos = _fragmentos_del_corpus()

    coleccion = abrir_coleccion(ruta_indice, embedding_function=embedder_passage)

    inicio = time.monotonic()
    resumen_indexacion = indexar_fragmentos(coleccion, fragmentos)
    tiempo_indexacion_s = time.monotonic() - inicio

    dimension = len(coleccion.peek(limit=1)["embeddings"][0])

    return {
        "nombre": nombre,
        "n_fragmentos": resumen_indexacion["total_en_coleccion"],
        "tiempo_indexacion_s": round(tiempo_indexacion_s, 2),
        "dimension_vector": dimension,
    }


def comparar(umbral_similitud: float = 0.88, api_proveedor: str = "gemini") -> pd.DataFrame:
    filas = []

    # --- Local (E5, Fase 2) ---
    ruta_local = RAIZ_PROYECTO / "data" / "index" / "comparacion_local"
    info_local = _construir_indice("local (multilingual-e5-small)", ruta_local, E5Embedder(modo="passage"))
    resultado_local = evaluar(ruta_local, E5Embedder(modo="query"), umbral_similitud)

    filas.append(
        {
            **info_local,
            "costo_indexacion_usd": 0.0,
            "costo_por_1000_consultas_usd": 0.0,
            **resultado_local["resumen"],
        }
    )

    # --- API ---
    ruta_api = RAIZ_PROYECTO / "data" / "index" / f"comparacion_{api_proveedor}"

    if api_proveedor == "openai":
        embedder_passage = OpenAIEmbedder(modo="passage")
        embedder_query = OpenAIEmbedder(modo="query")
        nombre_api = "API (text-embedding-3-small, OpenAI)"
    elif api_proveedor == "gemini":
        embedder_passage = GeminiEmbedder(modo="passage")  # espaciado sí (indexa cientos de fragmentos)
        embedder_query = GeminiEmbedder(modo="query", espaciar_llamadas=False)  # latencia real, sin freno artificial
        nombre_api = "API (gemini-embedding-2, capa gratuita)"
    else:
        raise ValueError(f"api_proveedor desconocido: {api_proveedor!r}")

    info_api = _construir_indice(nombre_api, ruta_api, embedder_passage)
    resultado_api = evaluar(ruta_api, embedder_query, umbral_similitud)

    if api_proveedor == "openai":
        tokens_indexacion = embedder_passage.tokens_acumulados
        tokens_consultas = embedder_query.tokens_acumulados
        n_consultas = resultado_api["resumen"]["n_preguntas_dominio"] + resultado_api["resumen"]["n_preguntas_fuera_dominio"]
        costo_indexacion = tokens_indexacion / 1_000_000 * PRECIO_OPENAI_USD_POR_1M_TOKENS
        costo_por_consulta = tokens_consultas / 1_000_000 * PRECIO_OPENAI_USD_POR_1M_TOKENS / n_consultas
        fuente, fecha = FUENTE_PRECIO_OPENAI, FECHA_VERIFICACION_PRECIO_OPENAI
    else:  # gemini: capa gratuita real -> $0, se deja la tarifa pagada de referencia como nota
        costo_indexacion = 0.0
        costo_por_consulta = 0.0
        fuente, fecha = FUENTE_PRECIO_GEMINI, FECHA_VERIFICACION_PRECIO_GEMINI

    filas.append(
        {
            **info_api,
            "costo_indexacion_usd": round(costo_indexacion, 6),
            "costo_por_1000_consultas_usd": round(costo_por_consulta * 1000, 6),
            **resultado_api["resumen"],
        }
    )

    tabla = pd.DataFrame(filas)

    reportes = RAIZ_PROYECTO / "data" / "outputs"
    reportes.mkdir(exist_ok=True)
    tabla.to_csv(reportes / "fase4_comparacion_embeddings.csv", index=False)
    resultado_local["detalle"].to_csv(reportes / "fase4_detalle_local.csv", index=False)
    resultado_api["detalle"].to_csv(reportes / f"fase4_detalle_{api_proveedor}.csv", index=False)

    print(f"\nFuente de precio API ({api_proveedor}): {fuente} (verificado {fecha})")
    if api_proveedor == "gemini":
        print(
            f"Tarifa pagada de referencia de Gemini (no cobrada, capa gratuita): "
            f"${PRECIO_GEMINI_USD_POR_1M_TOKENS_PAGADO} por 1M tokens"
        )

    return tabla


if __name__ == "__main__":
    tabla = comparar()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(tabla.to_string(index=False))

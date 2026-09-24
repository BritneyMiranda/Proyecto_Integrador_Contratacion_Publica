"""Proceso FUERA DE LÍNEA de la Tarea 1: PDF crudo -> índice vectorial listo.

Corre una sola vez (o cuando cambian los documentos fuente). El proceso EN
LÍNEA (`engine/motor.py`, usado por `app.py` e `src/interfaces/cli.py`) nunca
vuelve a leer los PDF ni reconstruye nada — solo abre lo que este script dejó
en `data/processed/` y `data/index/`.

Uso:
    .venv\\Scripts\\python.exe build_index.py

Requiere que los 2 PDF ya estén en `data/raw/` (ver README para cómo
conseguirlos — los sitios oficiales bloquean la descarga automática).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_PROYECTO / "src"))

from ingest.chunking import segmentar_paginas  # noqa: E402
from ingest.embeddings import crear_embedder  # noqa: E402
from ingest.informe_fase1 import generar_informe_fase1  # noqa: E402
from ingest.vectorstore import abrir_coleccion, indexar_fragmentos  # noqa: E402
from engine.motor import cargar_config  # noqa: E402


def main() -> None:
    config = cargar_config()
    inicio_total = time.monotonic()

    # ── Fase 1: verificar + limpiar los PDF -> data/processed/*.jsonl ──
    print("=== Fase 1: extracción y limpieza ===")
    tabla_calidad, _ = generar_informe_fase1(
        dir_salida_processed=RAIZ_PROYECTO / config["rutas"]["data_processed"],
        dir_reportes=RAIZ_PROYECTO / "data" / "outputs",
    )
    print(tabla_calidad.to_string(index=False))

    no_usables = tabla_calidad[tabla_calidad["usable"] != True]  # noqa: E712
    if not no_usables.empty:
        print(
            "\nADVERTENCIA: los siguientes documentos no se indexarán "
            f"(revisa data/raw/): {no_usables['doc_id'].tolist()}"
        )

    # ── Fase 2: chunking + embeddings + índice ──
    print("\n=== Fase 2: segmentación e indexación ===")
    dir_processed = RAIZ_PROYECTO / config["rutas"]["data_processed"]
    ruta_indice = RAIZ_PROYECTO / config["rutas"]["indice_vectorial"]
    proveedor_embeddings = config["embeddings"]["proveedor"]

    coleccion = abrir_coleccion(
        ruta_indice,
        embedding_function=crear_embedder(proveedor_embeddings, modo="passage"),
    )

    for ruta_jsonl in sorted(dir_processed.glob("*.jsonl")):
        paginas = [json.loads(linea) for linea in ruta_jsonl.read_text(encoding="utf-8").splitlines()]
        if not paginas:
            continue
        fragmentos = segmentar_paginas(
            paginas,
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"],
        )
        resumen = indexar_fragmentos(coleccion, fragmentos)
        print(f"{ruta_jsonl.stem}: {resumen}")

    print(f"\nÍndice final ({proveedor_embeddings}): {coleccion.count()} fragmentos en {ruta_indice}")
    print(f"Tiempo total: {time.monotonic() - inicio_total:.1f} s")


if __name__ == "__main__":
    main()

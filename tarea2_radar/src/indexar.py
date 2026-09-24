r"""Fase 3 de la Tarea 2 — construir el índice vectorial híbrido.

Reutiliza LITERALMENTE el código de embeddings e índice de la Tarea 1 (no
una copia): `E5Embedder`, `abrir_coleccion` e `indexar_fragmentos` se
importan directo desde `tarea1_rag_normativo/src/ingest`. Mismo modelo local
(`intfloat/multilingual-e5-small`), misma lógica de idempotencia por lote.

Cada proceso de contratación (una fila de `procesos_validados.parquet`) se
indexa como UN fragmento: el texto que se embebe es `titulo + descripcion`
(máx. 542 caracteres en este corpus — muy por debajo del límite de 512
tokens de E5, no hace falta partirlo). Los campos estructurados
(departamento, monto, categoría, fecha, comprador, ocid) se guardan como
METADATA de ChromaDB, no como texto — así se pueden usar como FILTROS
exactos en la consulta (`coleccion.query(..., where=...)`), separados de la
búsqueda semántica. El `id` de cada fragmento en Chroma es el propio `ocid`
del proceso (ya es único por diseño de OCDS).
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ_TAREA2 = Path(__file__).resolve().parents[1]
RAIZ_TAREA1 = RAIZ_TAREA2.parent / "tarea1_rag_normativo"
sys.path.insert(0, str(RAIZ_TAREA1 / "src"))
sys.path.insert(0, str(RAIZ_TAREA2 / "src"))

import pandas as pd

from ingest.embeddings import E5Embedder  # reusado de Tarea 1
from ingest.vectorstore import abrir_coleccion, indexar_fragmentos  # reusado de Tarea 1

DIR_PROCESSED = RAIZ_TAREA2 / "data" / "processed"
RUTA_INDICE = RAIZ_TAREA2 / "data" / "index" / "chroma_storage"


def _metadata_limpia(fila: pd.Series) -> dict:
    """ChromaDB solo acepta str/int/float/bool en metadata — nunca None/NaN."""
    def limpio(valor, por_defecto=""):
        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            return por_defecto
        return valor

    return {
        "ocid": fila["ocid"],
        "departamento": limpio(fila["departamento_normalizado"], "SIN_DEPARTAMENTO"),
        "monto": float(limpio(fila["monto"], 0.0)),
        "categoria": limpio(fila["categoria"], "sin_categoria"),
        "fecha_publicacion": limpio(fila["fecha_publicacion"], ""),
        "comprador_nombre": limpio(fila["comprador_nombre"], ""),
        "comprador_id": limpio(fila["comprador_id"], ""),
        "tender_id": limpio(fila["tender_id"], ""),
        "titulo": limpio(fila["titulo"], ""),
    }


def construir_fragmentos() -> list[dict]:
    df = pd.read_parquet(DIR_PROCESSED / "procesos_validados.parquet")
    fragmentos = []
    for _, fila in df.iterrows():
        texto = f"{fila['titulo'] or ''}. {fila['descripcion'] or ''}".strip()
        if not texto or texto == ".":
            continue
        fragmentos.append(
            {
                "chunk_id": fila["ocid"],
                "texto": texto,
                **_metadata_limpia(fila),
            }
        )
    return fragmentos


def _indexar_fragmentos_t2(coleccion, fragmentos: list[dict]) -> dict:
    """Variante de `indexar_fragmentos` de Tarea 1 adaptada a la metadata de
    procesos de contratación (la de Tarea 1 asume doc_id/pagina/version de
    normativa) — misma lógica de idempotencia (chequea qué ids ya existen
    antes de embeber), reimplementada aquí porque el esquema de metadata es
    distinto."""
    ids_solicitados = [f["chunk_id"] for f in fragmentos]
    existentes: set[str] = set()
    TAMANO_LOTE = 64
    for i in range(0, len(ids_solicitados), TAMANO_LOTE):
        lote = ids_solicitados[i : i + TAMANO_LOTE]
        existentes.update(coleccion.get(ids=lote, include=[])["ids"])

    pendientes = [f for f in fragmentos if f["chunk_id"] not in existentes]

    for i in range(0, len(pendientes), TAMANO_LOTE):
        lote = pendientes[i : i + TAMANO_LOTE]
        coleccion.add(
            ids=[f["chunk_id"] for f in lote],
            documents=[f["texto"] for f in lote],
            metadatas=[
                {k: v for k, v in f.items() if k not in ("chunk_id", "texto")} for f in lote
            ],
        )

    return {
        "fragmentos_recibidos": len(fragmentos),
        "ya_existian": len(existentes),
        "agregados_ahora": len(pendientes),
        "total_en_coleccion": coleccion.count(),
    }


if __name__ == "__main__":
    import time

    fragmentos = construir_fragmentos()
    print(f"Fragmentos a indexar: {len(fragmentos)}")

    coleccion = abrir_coleccion(RUTA_INDICE, embedding_function=E5Embedder(modo="passage"))

    inicio = time.monotonic()
    resumen = _indexar_fragmentos_t2(coleccion, fragmentos)
    duracion = time.monotonic() - inicio

    print(f"Resumen: {resumen}")
    print(f"Tiempo de indexación: {duracion:.1f} s")

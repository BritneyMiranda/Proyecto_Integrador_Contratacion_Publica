"""Índice vectorial persistente (ChromaDB) — idempotente y reanudable.

Diseño para cumplir los tres requisitos de Fase 2:

1. Idempotente: antes de embeber/agregar un lote, se consulta qué
   `chunk_id` ya existen en la colección (`collection.get(ids=...)`) y
   solo se procesan los que faltan. Correr `indexar_fragmentos` dos veces
   con la misma entrada dos veces no agrega nada la segunda vez, porque
   todos los ids ya están.
2. Reanudable: se procesa por lotes y cada lote se escribe a disco
   (ChromaDB `PersistentClient`) antes de pasar al siguiente. Si el
   proceso se interrumpe a la mitad, la siguiente corrida vuelve a
   consultar qué ids faltan y sigue exactamente donde se quedó — no
   recalcula embeddings ya guardados.
3. Aislamiento entre documentos: los `chunk_id` llevan el `doc_id` como
   prefijo (ver `chunking.py`), así que indexar un documento nuevo nunca
   puede chocar ni pisar los ids de otro. `indexar_fragmentos` nunca borra
   ni toca ids fuera de los que recibe en `fragmentos`.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb import Collection, EmbeddingFunction

from .embeddings import E5Embedder

TAMANO_LOTE = 32


def abrir_coleccion(
    ruta_almacen: str | Path,
    nombre_coleccion: str = "normativa_contrataciones",
    embedding_function: EmbeddingFunction | None = None,
) -> Collection:
    """`embedding_function` por defecto es el embedder local (E5). Se puede
    pasar otro (ej. el de OpenAI en Fase 4) para construir un índice
    paralelo con los mismos fragmentos pero otro modelo — ver
    `src/ingest/embeddings.py::crear_embedder`."""
    cliente = chromadb.PersistentClient(path=str(ruta_almacen))
    return cliente.get_or_create_collection(
        name=nombre_coleccion,
        embedding_function=embedding_function or E5Embedder(modo="passage"),
        metadata={"hnsw:space": "cosine"},
    )


def _ids_existentes(coleccion: Collection, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    existentes: set[str] = set()
    for i in range(0, len(ids), TAMANO_LOTE):
        lote = ids[i : i + TAMANO_LOTE]
        resultado = coleccion.get(ids=lote, include=[])
        existentes.update(resultado["ids"])
    return existentes


def indexar_fragmentos(coleccion: Collection, fragmentos: list[dict]) -> dict:
    """Agrega los fragmentos que falten. Devuelve un resumen para poder
    demostrar idempotencia/reanudación en el notebook."""
    ids_solicitados = [f["chunk_id"] for f in fragmentos]
    ya_existian = _ids_existentes(coleccion, ids_solicitados)

    pendientes = [f for f in fragmentos if f["chunk_id"] not in ya_existian]

    for i in range(0, len(pendientes), TAMANO_LOTE):
        lote = pendientes[i : i + TAMANO_LOTE]
        coleccion.add(
            ids=[f["chunk_id"] for f in lote],
            documents=[f["texto"] for f in lote],
            metadatas=[
                {
                    "doc_id": f["doc_id"],
                    "titulo": f["titulo"],
                    "institucion": f["institucion"],
                    "tipo": f["tipo"],
                    "version": f["version"],
                    "pagina": f["pagina"],
                    "indice_en_pagina": f["indice_en_pagina"],
                    "n_tokens": f["n_tokens"],
                }
                for f in lote
            ],
        )

    return {
        "fragmentos_recibidos": len(fragmentos),
        "ya_existian": len(ya_existian),
        "agregados_ahora": len(pendientes),
        "total_en_coleccion": coleccion.count(),
    }

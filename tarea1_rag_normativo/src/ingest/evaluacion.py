"""Compara configuraciones de chunking (chunk_size/chunk_overlap) contra
un conjunto de evaluación (eval/eval_fase2.yaml) usando recall@k: para
cada pregunta, ¿alguno de los k fragmentos recuperados cae en una de las
páginas esperadas del documento correcto?

Cada configuración se indexa en una colección ChromaDB EFÍMERA (en
memoria, se descarta al terminar) para no interferir con el índice
persistente real que se arma con la configuración ganadora.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import chromadb
import pandas as pd
import yaml

from .chunking import segmentar_paginas
from .embeddings import E5Embedder

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]


def cargar_preguntas_eval(
    ruta_yaml: str | Path = RAIZ_PROYECTO / "eval" / "eval_fase2.yaml",
) -> list[dict]:
    with open(ruta_yaml, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["preguntas"]


def _cargar_paginas_procesadas(dir_processed: str | Path) -> list[dict]:
    import json

    dir_processed = Path(dir_processed)
    registros = []
    for ruta in sorted(dir_processed.glob("*.jsonl")):
        registros.extend(json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines())
    return registros


def evaluar_config(
    chunk_size: int,
    chunk_overlap: int,
    k: int = 5,
    dir_processed: str | Path = RAIZ_PROYECTO / "data" / "processed",
    ruta_preguntas: str | Path = RAIZ_PROYECTO / "eval" / "eval_fase2.yaml",
) -> dict:
    paginas = _cargar_paginas_procesadas(dir_processed)
    fragmentos = segmentar_paginas(paginas, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    cliente_efimero = chromadb.Client()  # en memoria, no persiste
    nombre = f"eval_cs{chunk_size}_ov{chunk_overlap}"
    coleccion = cliente_efimero.get_or_create_collection(
        name=nombre,
        embedding_function=E5Embedder(modo="passage"),
        metadata={"hnsw:space": "cosine"},
    )
    coleccion.add(
        ids=[f["chunk_id"] for f in fragmentos],
        documents=[f["texto"] for f in fragmentos],
        metadatas=[{"doc_id": f["doc_id"], "pagina": f["pagina"]} for f in fragmentos],
    )

    embedder_query = E5Embedder(modo="query")
    preguntas = cargar_preguntas_eval(ruta_preguntas)

    detalle = []
    for p in preguntas:
        vector = embedder_query([p["pregunta"]])[0]
        resultados = coleccion.query(query_embeddings=[vector], n_results=k)
        metadatas = resultados["metadatas"][0]
        acierto = any(
            m["doc_id"] == p["doc_id"] and m["pagina"] in p["paginas_esperadas"]
            for m in metadatas
        )
        detalle.append(
            {
                "pregunta": p["pregunta"],
                "doc_id": p["doc_id"],
                "paginas_esperadas": p["paginas_esperadas"],
                "paginas_recuperadas": [m["pagina"] for m in metadatas],
                "acierto": acierto,
            }
        )

    recall_at_k = sum(d["acierto"] for d in detalle) / len(detalle)

    longitudes = [f["n_tokens"] for f in fragmentos]
    por_doc = pd.DataFrame(fragmentos).groupby("doc_id").size().to_dict()

    cliente_efimero.delete_collection(nombre)

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": k,
        "recall_at_k": round(recall_at_k, 3),
        "n_preguntas": len(detalle),
        "n_fragmentos_total": len(fragmentos),
        "fragmentos_por_doc": por_doc,
        "longitud_tokens_min": min(longitudes),
        "longitud_tokens_mediana": statistics.median(longitudes),
        "longitud_tokens_media": round(statistics.mean(longitudes), 1),
        "longitud_tokens_p90": statistics.quantiles(longitudes, n=10)[8],
        "longitud_tokens_max": max(longitudes),
        "detalle": detalle,
    }

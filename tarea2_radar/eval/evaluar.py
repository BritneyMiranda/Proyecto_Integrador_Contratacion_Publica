r"""Fase 3 de la Tarea 2 — recall@k sobre `eval/preguntas.csv`, SIN llamar
al LLM (mismo principio que `tarea1_rag_normativo/eval/evaluar.py`: mide
solo la etapa de recuperación — embedder + filtros + índice —, que es
gratis, determinística y rápida de iterar).

A diferencia de la Tarea 1 (recall@k por doc_id+página), aquí un acierto es
que el `ocid` esperado aparezca entre los k resultados devueltos por
`coleccion.query()` — el `ocid` es el identificador único del proceso
(ver `src/procesar.py` para la explicación completa release vs. record).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

RAIZ_TAREA2 = Path(__file__).resolve().parents[1]
RAIZ_TAREA1 = RAIZ_TAREA2.parent / "tarea1_rag_normativo"
sys.path.insert(0, str(RAIZ_TAREA1 / "src"))
sys.path.insert(0, str(RAIZ_TAREA2 / "src"))

import pandas as pd

from ingest.embeddings import E5Embedder
from ingest.vectorstore import abrir_coleccion

from filtros import extraer_filtros

RUTA_INDICE = RAIZ_TAREA2 / "data" / "index" / "chroma_storage"


def cargar_preguntas(ruta: Path = RAIZ_TAREA2 / "eval" / "preguntas.csv") -> pd.DataFrame:
    return pd.read_csv(ruta)


def evaluar(
    umbral_similitud: float,
    alias_categoria: dict[str, str],
    ruta_preguntas: Path = RAIZ_TAREA2 / "eval" / "preguntas.csv",
    k_max: int = 5,
    usar_filtros: bool = True,
) -> dict:
    preguntas = cargar_preguntas(ruta_preguntas)
    coleccion = abrir_coleccion(RUTA_INDICE, embedding_function=E5Embedder(modo="passage"))
    embedder_query = E5Embedder(modo="query")

    detalle = []
    for _, fila in preguntas.iterrows():
        inicio = time.monotonic()
        where = extraer_filtros(fila["pregunta"], alias_categoria) if usar_filtros else None
        vector = embedder_query([fila["pregunta"]])[0]
        argumentos = {"query_embeddings": [vector], "n_results": k_max}
        if where:
            argumentos["where"] = where
        resultados = coleccion.query(**argumentos)
        latencia_ms = (time.monotonic() - inicio) * 1000

        ocids_recuperados = [m["ocid"] for m in resultados["metadatas"][0]] if resultados["ids"][0] else []
        distancias = resultados["distances"][0] if resultados["ids"][0] else []
        mejor_similitud = 1 - distancias[0] if distancias else 0.0
        abstuvo = mejor_similitud < umbral_similitud

        aciertos = {k: (fila["ocid_esperado"] in ocids_recuperados[:k]) for k in (1, 3, 5) if k <= k_max}

        detalle.append(
            {
                "id": fila["id"],
                "pregunta": fila["pregunta"],
                "ocid_esperado": fila["ocid_esperado"],
                "filtros_aplicados": where,
                "mejor_similitud": round(mejor_similitud, 4),
                "abstuvo": abstuvo,
                "latencia_ms": round(latencia_ms, 1),
                "acierto_top1": aciertos.get(1),
                "acierto_top3": aciertos.get(3),
                "acierto_top5": aciertos.get(5),
            }
        )

    tabla = pd.DataFrame(detalle)
    resumen = {
        "n_preguntas": len(tabla),
        "recall_at_1": round(tabla["acierto_top1"].mean(), 3),
        "recall_at_3": round(tabla["acierto_top3"].mean(), 3),
        "recall_at_5": round(tabla["acierto_top5"].mean(), 3),
        "abstenciones": int(tabla["abstuvo"].sum()),
        "latencia_ms_promedio": round(tabla["latencia_ms"].mean(), 1),
        "preguntas_con_filtro": int(tabla["filtros_aplicados"].notna().sum()),
    }
    return {"detalle": tabla, "resumen": resumen}


if __name__ == "__main__":
    import yaml

    with open(RAIZ_TAREA2 / "config.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    resultado = evaluar(
        umbral_similitud=config["recuperacion"]["umbral_similitud"],
        alias_categoria=config["filtros"]["alias_categoria"],
    )

    print("=== Resumen recall@k (Tarea 2) ===")
    for clave, valor in resultado["resumen"].items():
        print(f"{clave}: {valor}")

    print("\n=== Detalle ===")
    print(resultado["detalle"][["id", "pregunta", "filtros_aplicados", "mejor_similitud", "acierto_top5"]].to_string(index=False))

    dir_reportes = RAIZ_TAREA2 / "data" / "outputs"
    dir_reportes.mkdir(exist_ok=True)
    resultado["detalle"].to_csv(dir_reportes / "fase3_evaluacion.csv", index=False)
    pd.DataFrame([resultado["resumen"]]).to_csv(dir_reportes / "fase3_resumen.csv", index=False)
    print(f"\nGuardado en {dir_reportes}")

r"""Fase 3 — ¿el umbral de la Tarea 1 (0.88) sirve para este corpus?

Reutiliza LITERALMENTE `barrer_umbrales` de
`tarea1_rag_normativo/src/engine/calibracion.py` (esa función es genérica:
solo necesita una tabla con columnas `mejor_similitud` y
`deberia_abstenerse`, no sabe nada de normativa ni de contratación). Lo
único propio de esta tarea es cómo se arma esa tabla: preguntas de
`eval/preguntas.csv` (dominio) + `eval/eval_abstencion.yaml` (fuera de
dominio), contra el índice de procesos de contratación.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ_TAREA2 = Path(__file__).resolve().parents[1]
RAIZ_TAREA1 = RAIZ_TAREA2.parent / "tarea1_rag_normativo"
sys.path.insert(0, str(RAIZ_TAREA1 / "src"))
sys.path.insert(0, str(RAIZ_TAREA2 / "src"))

import pandas as pd
import yaml

from ingest.embeddings import E5Embedder
from ingest.vectorstore import abrir_coleccion

from engine.calibracion import barrer_umbrales  # reusado literal de Tarea 1

RUTA_INDICE = RAIZ_TAREA2 / "data" / "index" / "chroma_storage"


def calcular_similitudes() -> pd.DataFrame:
    coleccion = abrir_coleccion(RUTA_INDICE, embedding_function=E5Embedder(modo="passage"))
    embedder_query = E5Embedder(modo="query")

    preguntas_dentro = pd.read_csv(RAIZ_TAREA2 / "eval" / "preguntas.csv")["pregunta"].tolist()
    with open(RAIZ_TAREA2 / "eval" / "eval_abstencion.yaml", encoding="utf-8") as fh:
        preguntas_fuera = [p["pregunta"] for p in yaml.safe_load(fh)["fuera_del_corpus"]]

    filas = [{"pregunta": p, "deberia_abstenerse": False} for p in preguntas_dentro]
    filas += [{"pregunta": p, "deberia_abstenerse": True} for p in preguntas_fuera]

    for fila in filas:
        vector = embedder_query([fila["pregunta"]])[0]
        resultado = coleccion.query(query_embeddings=[vector], n_results=1)
        fila["mejor_similitud"] = round(1 - resultado["distances"][0][0], 4)

    return pd.DataFrame(filas)


if __name__ == "__main__":
    tabla = calcular_similitudes()
    print("=== Similitudes ===")
    print(tabla.sort_values("mejor_similitud").to_string(index=False))

    candidatos = [round(x, 2) for x in [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92]]
    barrido = barrer_umbrales(tabla, candidatos)
    print("\n=== Barrido de umbrales ===")
    print(barrido.to_string(index=False))

    print("\n--- Umbral de la Tarea 1 (0.88) aplicado a ESTE corpus ---")
    fila_088 = barrido[barrido["umbral"] == 0.88].iloc[0]
    print(fila_088.to_dict())

    dir_reportes = RAIZ_TAREA2 / "data" / "outputs"
    dir_reportes.mkdir(exist_ok=True)
    tabla.to_csv(dir_reportes / "fase3_similitudes_calibracion.csv", index=False)
    barrido.to_csv(dir_reportes / "fase3_barrido_umbral.csv", index=False)

"""Barrido de umbral de abstención sobre eval/eval_abstencion.yaml.

Para cada pregunta (dentro o fuera del corpus) se calcula la similitud
coseno del MEJOR fragmento recuperado (top-1) contra el índice real. Con
esas similitudes ya calculadas una sola vez, se prueban muchos umbrales
sin volver a llamar al modelo de embeddings — el barrido es barato.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ingest.embeddings import E5Embedder
from ingest.vectorstore import abrir_coleccion

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]


def _cargar_preguntas_dentro(ruta: Path) -> list[str]:
    with open(ruta, encoding="utf-8") as fh:
        return [p["pregunta"] for p in yaml.safe_load(fh)["preguntas"]]


def _cargar_preguntas_fuera(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["fuera_del_corpus"]


def calcular_similitudes(
    ruta_indice: str | Path,
    ruta_eval_fase2: str | Path = RAIZ_PROYECTO / "eval" / "eval_fase2.yaml",
    ruta_eval_abstencion: str | Path = RAIZ_PROYECTO / "eval" / "eval_abstencion.yaml",
    k: int = 1,
) -> pd.DataFrame:
    """Devuelve una tabla con una fila por pregunta: pregunta,
    deberia_abstenerse, mejor_similitud."""
    coleccion = abrir_coleccion(ruta_indice)
    embedder_query = E5Embedder(modo="query")

    filas = []
    for pregunta in _cargar_preguntas_dentro(ruta_eval_fase2):
        filas.append({"pregunta": pregunta, "deberia_abstenerse": False})
    for item in _cargar_preguntas_fuera(ruta_eval_abstencion):
        filas.append({"pregunta": item["pregunta"], "deberia_abstenerse": True, "motivo": item.get("motivo", "")})

    for fila in filas:
        vector = embedder_query([fila["pregunta"]])[0]
        resultado = coleccion.query(query_embeddings=[vector], n_results=k)
        distancia = resultado["distances"][0][0]
        fila["mejor_similitud"] = round(1 - distancia, 4)

    return pd.DataFrame(filas)


def barrer_umbrales(tabla_similitudes: pd.DataFrame, candidatos: list[float]) -> pd.DataFrame:
    """Para cada umbral candidato, calcula:
    - falsos_abstencion: preguntas CON respuesta en el corpus que el
      umbral haría abstenerse igual (oportunidad perdida).
    - falsos_respuesta: preguntas SIN respuesta en el corpus que el
      umbral dejaría pasar al LLM (riesgo de alucinar con confianza).
    """
    filas = []
    dentro = tabla_similitudes[~tabla_similitudes["deberia_abstenerse"]]
    fuera = tabla_similitudes[tabla_similitudes["deberia_abstenerse"]]

    for umbral in candidatos:
        abstiene_si_umbral = tabla_similitudes["mejor_similitud"] < umbral

        falsos_abstencion = int((abstiene_si_umbral & ~tabla_similitudes["deberia_abstenerse"]).sum())
        falsos_respuesta = int((~abstiene_si_umbral & tabla_similitudes["deberia_abstenerse"]).sum())

        filas.append(
            {
                "umbral": umbral,
                "falsos_abstencion": falsos_abstencion,
                "falsos_abstencion_pct": round(falsos_abstencion / len(dentro) * 100, 1) if len(dentro) else 0.0,
                "falsos_respuesta": falsos_respuesta,
                "falsos_respuesta_pct": round(falsos_respuesta / len(fuera) * 100, 1) if len(fuera) else 0.0,
                "errores_totales": falsos_abstencion + falsos_respuesta,
            }
        )
    return pd.DataFrame(filas)

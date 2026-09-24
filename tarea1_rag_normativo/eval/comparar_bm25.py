"""Innovación (bono, no forma parte de los 20 puntos base): BM25 (palabras
clave) vs. búsqueda semántica (embeddings) sobre las 5 preguntas de estilo
COLOQUIAL de `eval/preguntas.csv` (ids 17-21).

Motivación: la búsqueda semántica entiende paráfrasis y sinónimos ("plata
por adelantado" ~ "adelanto directo al contratista"), mientras que BM25 es
un algoritmo puramente LÉXICO — cuenta términos compartidos, ponderados por
frecuencia inversa de documento (TF-IDF probabilístico). Dos formas de
preguntar lo mismo que no comparten palabras pueden fallar en BM25 y acertar
en embeddings. Esta comparación mide esa diferencia con los MISMOS 751
fragmentos que ya indexa `build_index.py` (se leen tal cual del índice
ChromaDB ya construido — no se reconstruye nada, no se llama al LLM).

Limitaciones de esta comparación (documentadas, no escondidas):
- Solo 5 preguntas (las únicas marcadas `estilo=coloquial`) — evidencia
  puntual, no una conclusión estadísticamente robusta ni generalizable.
- Tokenización simple (minúsculas + separar por caracteres alfanuméricos,
  sin stopwords ni stemming en español) — una tokenización más sofisticada
  podría mover el resultado.
- El score de BM25 no es una probabilidad ni es comparable en escala directa
  con la similitud coseno de embeddings; aquí solo se usa para RANKEAR
  (top-k), no para replicar el umbral de abstención de Fase 3.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PROYECTO / "src"))

from ingest.vectorstore import abrir_coleccion  # noqa: E402
from engine.motor import cargar_config  # noqa: E402

_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+")


def _tokenizar(texto: str) -> list[str]:
    return _TOKEN_RE.findall(texto.lower())


def _cargar_fragmentos_del_indice(ruta_indice: Path) -> tuple[list[str], list[dict]]:
    """Lee los fragmentos YA indexados por `build_index.py` (mismo corpus
    que usan los embeddings, ni un fragmento de más ni de menos) — abre el
    índice existente, nunca lo reconstruye."""
    coleccion = abrir_coleccion(ruta_indice)
    datos = coleccion.get(limit=coleccion.count(), include=["documents", "metadatas"])
    return datos["documents"], datos["metadatas"]


def comparar_bm25_vs_semantica(k_max: int = 5) -> tuple[pd.DataFrame, dict]:
    config = cargar_config()
    ruta_indice = RAIZ_PROYECTO / config["rutas"]["indice_vectorial"]

    documentos, metadatas = _cargar_fragmentos_del_indice(ruta_indice)
    corpus_tokenizado = [_tokenizar(d) for d in documentos]
    bm25 = BM25Okapi(corpus_tokenizado)

    preguntas = pd.read_csv(RAIZ_PROYECTO / "eval" / "preguntas.csv", keep_default_na=False)
    coloquiales = preguntas[preguntas["estilo"] == "coloquial"].copy()
    coloquiales["paginas_esperadas"] = coloquiales["paginas_esperadas"].apply(
        lambda s: [int(p) for p in s.split(";")] if s else []
    )

    # Resultados semánticos (Fase 4) ya calculados por eval/evaluar.py — se
    # reutilizan tal cual, sin recargar el modelo de embeddings ni recalcular.
    ruta_semantica = RAIZ_PROYECTO / "data" / "outputs" / "fase4_evaluacion_local.csv"
    semantica = pd.read_csv(ruta_semantica).set_index("id")

    filas = []
    for _, fila in coloquiales.iterrows():
        objetivo = {(fila["doc_id_esperado"], p) for p in fila["paginas_esperadas"]}

        tokens_query = _tokenizar(fila["pregunta"])
        scores = bm25.get_scores(tokens_query)
        orden = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_max]

        aciertos_bm25 = {}
        for k in (1, 3, 5):
            recuperado = {(metadatas[i]["doc_id"], metadatas[i]["pagina"]) for i in orden[:k]}
            aciertos_bm25[k] = bool(objetivo & recuperado)

        fila_semantica = semantica.loc[fila["id"]]

        filas.append(
            {
                "id": fila["id"],
                "pregunta": fila["pregunta"],
                "bm25_mejor_score": round(float(scores[orden[0]]), 3) if orden else 0.0,
                "bm25_acierto_top1": aciertos_bm25[1],
                "bm25_acierto_top3": aciertos_bm25[3],
                "bm25_acierto_top5": aciertos_bm25[5],
                "semantica_mejor_similitud": fila_semantica["mejor_similitud"],
                "semantica_acierto_top1": bool(fila_semantica["acierto_top1"]),
                "semantica_acierto_top3": bool(fila_semantica["acierto_top3"]),
                "semantica_acierto_top5": bool(fila_semantica["acierto_top5"]),
            }
        )

    detalle = pd.DataFrame(filas)

    resumen = {
        "n_preguntas": len(detalle),
        "bm25_recall_at_1": round(detalle["bm25_acierto_top1"].mean(), 3),
        "bm25_recall_at_3": round(detalle["bm25_acierto_top3"].mean(), 3),
        "bm25_recall_at_5": round(detalle["bm25_acierto_top5"].mean(), 3),
        "semantica_recall_at_1": round(detalle["semantica_acierto_top1"].mean(), 3),
        "semantica_recall_at_3": round(detalle["semantica_acierto_top3"].mean(), 3),
        "semantica_recall_at_5": round(detalle["semantica_acierto_top5"].mean(), 3),
    }

    dir_salida = RAIZ_PROYECTO / "data" / "outputs"
    dir_salida.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(dir_salida / "fase4_bm25_vs_semantica.csv", index=False)
    pd.DataFrame([resumen]).to_csv(dir_salida / "fase4_bm25_vs_semantica_resumen.csv", index=False)

    return detalle, resumen


if __name__ == "__main__":
    detalle, resumen = comparar_bm25_vs_semantica()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(detalle.to_string(index=False))
    print("\n=== Resumen (5 preguntas coloquiales) ===")
    for clave, valor in resumen.items():
        print(f"{clave}: {valor}")

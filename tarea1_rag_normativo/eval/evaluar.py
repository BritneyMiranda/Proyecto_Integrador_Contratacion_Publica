"""Evaluación de recuperación y abstención — NUNCA llama al modelo de generación.

Mide dos etapas DISTINTAS del pipeline, por separado:

1. Recall@1 / Recall@3 / Recall@5 — evalúa la etapa de RECUPERACIÓN (el
   embedder + el índice vectorial). Para cada pregunta de dominio, ¿el
   fragmento correcto (doc_id + página esperada) aparece entre los k
   primeros resultados de `coleccion.query()`? No importa si el LLM
   redactaría bien o mal la respuesta final: esto solo mide si la
   información correcta LLEGARÍA al contexto que vería el LLM.

2. Tasa de abstención (correcta / incorrecta) — evalúa la etapa de DECISIÓN
   (el umbral de similitud calibrado en Fase 3). Para cada pregunta, de
   dominio o fuera de dominio, ¿el motor decidiría correctamente si debe
   intentar responder o abstenerse, usando solo la similitud del mejor
   fragmento? Esto pasa DESPUÉS de la recuperación pero ANTES de llamar
   al LLM — exactamente donde `engine.motor.responder()` corta el flujo.

Por qué importa que esta evaluación no tenga costo: el modelo de embeddings,
el chunking (tamaño/overlap) y el umbral de abstención son justamente los
parámetros que más se ajustan durante el desarrollo — se corren decenas de
veces mientras se prueba una configuración nueva. Si cada corrida llamara al
LLM, iterar sería lento y costoso. Separar "¿la recuperación trae lo
correcto?" (gratis, determinístico, rápido — este script) de "¿la respuesta
generada por el LLM es buena?" (necesita el LLM, cuesta dinero, no
determinístico) permite optimizar exhaustivamente la parte barata y gastar
el presupuesto de generación solo en la validación final, con una
configuración ya elegida.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from chromadb import EmbeddingFunction

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]


def cargar_preguntas(ruta: str | Path = RAIZ_PROYECTO / "eval" / "preguntas.csv") -> pd.DataFrame:
    df = pd.read_csv(ruta, keep_default_na=False)
    df["paginas_esperadas"] = df["paginas_esperadas"].apply(
        lambda s: [int(p) for p in s.split(";")] if s else []
    )
    return df


def evaluar(
    ruta_indice: str | Path,
    embedder_query: EmbeddingFunction,
    umbral_similitud: float,
    ruta_preguntas: str | Path = RAIZ_PROYECTO / "eval" / "preguntas.csv",
    k_max: int = 5,
) -> dict:
    """No llama al modelo de generación en ningún momento. `embedder_query`
    debe ser el MISMO modelo con el que se construyó el índice en
    `ruta_indice` (local u OpenAI), para que la comparación sea justa."""
    from ingest.vectorstore import abrir_coleccion

    preguntas = cargar_preguntas(ruta_preguntas)
    coleccion = abrir_coleccion(ruta_indice)  # embedding_function no se usa: pasamos query_embeddings ya calculados

    detalle = []
    for _, fila in preguntas.iterrows():
        inicio = time.monotonic()
        vector = embedder_query([fila["pregunta"]])[0]
        resultados = coleccion.query(query_embeddings=[vector], n_results=k_max)
        latencia_ms = (time.monotonic() - inicio) * 1000

        metadatas = resultados["metadatas"][0]
        distancias = resultados["distances"][0]
        mejor_similitud = 1 - distancias[0] if distancias else 0.0
        abstuvo = mejor_similitud < umbral_similitud

        es_dominio = fila["ambito"] == "dominio"
        aciertos = {}
        if es_dominio:
            objetivo = {(fila["doc_id_esperado"], p) for p in fila["paginas_esperadas"]}
            for k in (1, 3, 5):
                if k > k_max:
                    continue
                recuperado_top_k = {(m["doc_id"], m["pagina"]) for m in metadatas[:k]}
                aciertos[k] = bool(objetivo & recuperado_top_k)

        detalle.append(
            {
                "id": fila["id"],
                "pregunta": fila["pregunta"],
                "ambito": fila["ambito"],
                "es_dominio": es_dominio,
                "mejor_similitud": round(mejor_similitud, 4),
                "abstuvo": abstuvo,
                "latencia_ms": round(latencia_ms, 1),
                "acierto_top1": aciertos.get(1),
                "acierto_top3": aciertos.get(3),
                "acierto_top5": aciertos.get(5),
            }
        )

    tabla = pd.DataFrame(detalle)
    dominio = tabla[tabla["es_dominio"]]
    fuera = tabla[~tabla["es_dominio"]]

    resumen = {
        "n_preguntas_dominio": len(dominio),
        "n_preguntas_fuera_dominio": len(fuera),
        "recall_at_1": round(dominio["acierto_top1"].mean(), 3) if len(dominio) else None,
        "recall_at_3": round(dominio["acierto_top3"].mean(), 3) if len(dominio) else None,
        "recall_at_5": round(dominio["acierto_top5"].mean(), 3) if len(dominio) else None,
        "abstenciones_correctas": int(fuera["abstuvo"].sum()) if len(fuera) else 0,
        "abstenciones_incorrectas": int(dominio["abstuvo"].sum()) if len(dominio) else 0,
        "tasa_abstencion_correcta_pct": round(fuera["abstuvo"].mean() * 100, 1) if len(fuera) else None,
        "tasa_abstencion_incorrecta_pct": round(dominio["abstuvo"].mean() * 100, 1) if len(dominio) else None,
        "latencia_ms_promedio_consulta": round(tabla["latencia_ms"].mean(), 1),
    }

    return {"detalle": tabla, "resumen": resumen}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(RAIZ_PROYECTO / "src"))
    from ingest.embeddings import E5Embedder

    RUTA_INDICE_PRODUCCION = RAIZ_PROYECTO / "data" / "index" / "chroma_storage"
    UMBRAL = 0.88  # el mismo de config.yaml (recuperacion.umbral_similitud)

    resultado = evaluar(RUTA_INDICE_PRODUCCION, E5Embedder(modo="query"), UMBRAL)

    print("=== Resumen (índice de producción, local) ===")
    for clave, valor in resultado["resumen"].items():
        print(f"{clave}: {valor}")

    dir_reportes = RAIZ_PROYECTO / "data" / "outputs"
    dir_reportes.mkdir(exist_ok=True)
    resultado["detalle"].to_csv(dir_reportes / "fase4_evaluacion_local.csv", index=False)
    pd.DataFrame([resultado["resumen"]]).to_csv(dir_reportes / "fase4_resumen.csv", index=False)
    print(f"\nDetalle guardado en {dir_reportes / 'fase4_evaluacion_local.csv'}")
    print(f"Resumen guardado en {dir_reportes / 'fase4_resumen.csv'}")

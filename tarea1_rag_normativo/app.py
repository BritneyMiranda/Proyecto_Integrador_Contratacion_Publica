"""Interfaz Streamlit del asistente RAG normativo (Fase 5).

Un solo archivo, como pide el enunciado. Sigue el mismo principio que
`src/interfaces/cli.py`: esta app NO tiene ninguna lógica de RAG propia, solo
llama a `engine.motor.responder()` y muestra el resultado. Nunca reconstruye
el índice vectorial al iniciarse — `abrir_coleccion()` (dentro de
`motor.py`) solo abre lo que ya existe en `data/index/chroma_storage/`; si
no hay nada indexado todavía, esta app lo dice explícitamente en vez de
disparar el pipeline de chunking/embeddings.

Correr con:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

RAIZ_PROYECTO = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_PROYECTO / "src"))

load_dotenv(RAIZ_PROYECTO / ".env")

from engine.motor import ErrorDeAPI, cargar_config, responder  # noqa: E402

st.set_page_config(page_title="Asistente RAG · Contratación Pública", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Cargando configuración y modelo de embeddings...")
def _config_cacheada() -> dict:
    return cargar_config()


def _leer_csv_si_existe(ruta: Path) -> pd.DataFrame | None:
    return pd.read_csv(ruta) if ruta.exists() else None


def _contar_fragmentos_indice(config: dict) -> int | None:
    """Solo ABRE el índice existente para contar cuántos fragmentos tiene —
    nunca lo construye ni le agrega nada. Devuelve None si la carpeta del
    índice no existe todavía (nada que abrir)."""
    ruta_indice = RAIZ_PROYECTO / config["rutas"]["indice_vectorial"]
    if not ruta_indice.exists():
        return None
    from ingest.vectorstore import abrir_coleccion

    return abrir_coleccion(ruta_indice).count()


config = _config_cacheada()

st.title("⚖️ Asistente RAG · Contratación Pública Peruana")
st.caption(
    "Ley N.° 32069 (Ley General de Contrataciones Públicas) + "
    "Decreto Supremo N.° 001-2026-EF (modifica el Reglamento)"
)

pestana_asistente, pestana_calidad = st.tabs(["💬 Preguntar", "📊 Calidad y evaluación"])

# ─────────────────────────────────────────────────────────────────────────
# Pestaña 1: el asistente
# ─────────────────────────────────────────────────────────────────────────
with pestana_asistente:
    n_fragmentos = _contar_fragmentos_indice(config)

    if n_fragmentos is None:
        st.warning(
            "No hay ningún índice en `data/index/chroma_storage/` todavía. "
            "Esta app nunca lo construye sola — corre primero `build_index.py`."
        )
    else:
        st.info(
            f"Índice cargado: **{n_fragmentos} fragmentos** — "
            f"embeddings `{config['embeddings']['proveedor']}`, "
            f"generación `{config['generacion']['proveedor']}` "
            f"({config['generacion']['modelo']}), "
            f"umbral de abstención `{config['recuperacion']['umbral_similitud']}`."
        )

    with st.form("formulario_pregunta"):
        pregunta = st.text_area(
            "Tu pregunta",
            placeholder="Ej: ¿Se necesita una adenda para aprobar una prestación adicional de obra?",
            height=90,
        )
        enviar = st.form_submit_button("Preguntar", disabled=(n_fragmentos is None))

    if enviar and pregunta.strip():
        with st.spinner("Recuperando fragmentos y generando la respuesta..."):
            try:
                resultado = responder(pregunta.strip(), config)
            except ErrorDeAPI as err:
                st.error(
                    f"**Error del proveedor de generación** (`{err.proveedor}`"
                    + (f", código {err.status_code}" if err.status_code else "")
                    + f"):\n\n{err}"
                )
                resultado = None

        if resultado is not None:
            if resultado["abstuvo"]:
                st.warning("⚠️ **El motor se abstuvo de responder** — la pregunta cae fuera del corpus indexado.")
            else:
                st.success("Respuesta generada con el contexto recuperado.")

            st.markdown("### Respuesta")
            st.write(resultado["respuesta"])

            columna_costo, columna_similitud, columna_estado = st.columns(3)
            columna_costo.metric(
                "Costo de la consulta",
                f"${resultado['costo_usd']:.6f}" if not resultado["abstuvo"] else "$0.00 (no llamó al LLM)",
            )
            columna_similitud.metric("Mejor similitud", f"{resultado['mejor_similitud']:.4f}")
            columna_estado.metric("¿Se abstuvo?", "Sí" if resultado["abstuvo"] else "No")

            if not resultado["abstuvo"]:
                st.caption(
                    f"tokens entrada: {resultado['tokens_entrada']} · "
                    f"tokens salida: {resultado['tokens_salida']} · "
                    f"franja horaria: {resultado['franja_horaria']} · "
                    f"latencia: {resultado['latencia_ms']:.0f} ms"
                )

            st.markdown("### Fragmentos citados")
            tabla_fuentes = pd.DataFrame(resultado["fuentes"])[
                ["titulo", "pagina", "version", "similitud"]
            ].rename(
                columns={
                    "titulo": "Documento",
                    "pagina": "Página",
                    "version": "Versión",
                    "similitud": "Similitud",
                }
            )
            st.dataframe(tabla_fuentes, width='stretch', hide_index=True)
    elif enviar:
        st.error("Escribe una pregunta antes de enviar.")

# ─────────────────────────────────────────────────────────────────────────
# Pestaña 2: calidad de extracción (Fase 1) + evaluación (Fase 4)
# ─────────────────────────────────────────────────────────────────────────
with pestana_calidad:
    st.subheader("Fase 1 — Calidad de extracción")
    tabla_fase1 = _leer_csv_si_existe(RAIZ_PROYECTO / "data" / "outputs" / "fase1_calidad.csv")
    if tabla_fase1 is not None:
        st.dataframe(tabla_fase1, width='stretch', hide_index=True)
    else:
        st.caption("Todavía no existe `data/outputs/fase1_calidad.csv` — corre `build_index.py`.")

    st.subheader("Fase 4 — Evaluación de recuperación y abstención")
    resumen_fase4 = _leer_csv_si_existe(RAIZ_PROYECTO / "data" / "outputs" / "fase4_resumen.csv")
    if resumen_fase4 is not None:
        fila = resumen_fase4.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall@1", fila["recall_at_1"])
        c2.metric("Recall@3", fila["recall_at_3"])
        c3.metric("Recall@5", fila["recall_at_5"])
        c4.metric("Latencia prom. consulta", f"{fila['latencia_ms_promedio_consulta']:.0f} ms")
        c5, c6 = st.columns(2)
        c5.metric(
            "Abstenciones correctas",
            f"{fila['abstenciones_correctas']}/{fila['n_preguntas_fuera_dominio']}",
            help="Preguntas fuera del corpus donde el motor sí se abstuvo.",
        )
        c6.metric(
            "Abstenciones incorrectas",
            f"{fila['abstenciones_incorrectas']}/{fila['n_preguntas_dominio']}",
            help="Preguntas del corpus donde el motor se abstuvo sin deber hacerlo.",
        )
    else:
        st.caption("Todavía no existe `data/outputs/fase4_resumen.csv` — corre `python eval/evaluar.py`.")

    detalle_fase4 = _leer_csv_si_existe(RAIZ_PROYECTO / "data" / "outputs" / "fase4_evaluacion_local.csv")
    if detalle_fase4 is not None:
        with st.expander("Ver detalle pregunta por pregunta (eval/preguntas.csv)"):
            st.dataframe(detalle_fase4, width='stretch', hide_index=True)

    comparacion_embeddings = _leer_csv_si_existe(RAIZ_PROYECTO / "data" / "outputs" / "fase4_comparacion_embeddings.csv")
    if comparacion_embeddings is not None:
        st.subheader("Fase 4 — Comparación de embeddings (local vs. API)")
        st.dataframe(comparacion_embeddings, width='stretch', hide_index=True)

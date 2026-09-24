"""Panel Streamlit del radar de contratación pública (Tarea 2, Fase 4).

Un solo archivo. Lee ÚNICAMENTE archivos ya precalculados por las Fases
1-3 (`data/processed/procesos_validados.parquet`, `.../departamentos.geojson`,
`data/outputs/fase1_calidad.csv`, `data/outputs/fase2_calidad_validacion.csv`) — nunca
descarga archivos de OECE ni reconstruye el índice vectorial al cargar la
página. La única llamada "en vivo" es al motor RAG (`src/motor.py`), que a
su vez solo ABRE el índice ya construido (`src/indexar.py`), no lo crea.

Correr con:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

RAIZ_TAREA2 = Path(__file__).resolve().parent
RAIZ_TAREA1 = RAIZ_TAREA2.parent / "tarea1_rag_normativo"
sys.path.insert(0, str(RAIZ_TAREA1 / "src"))
sys.path.insert(0, str(RAIZ_TAREA2 / "src"))

load_dotenv(RAIZ_TAREA2 / ".env")

from motor import ErrorDeAPI, cargar_config, responder  # noqa: E402

st.set_page_config(page_title="Radar de Contratación Pública · Perú", page_icon="📡", layout="wide")

DEPARTAMENTOS_OFICIALES = [
    "AMAZONAS", "ANCASH", "APURÍMAC", "AREQUIPA", "AYACUCHO", "CAJAMARCA",
    "CALLAO", "CUSCO", "HUANCAVELICA", "HUÁNUCO", "ICA", "JUNÍN",
    "LA LIBERTAD", "LAMBAYEQUE", "LIMA", "LORETO", "MADRE DE DIOS",
    "MOQUEGUA", "PASCO", "PIURA", "PUNO", "SAN MARTÍN", "TACNA", "TUMBES",
    "UCAYALI",
]


# ─────────────────────────────────────────────────────────────────────────
# Carga de archivos PRECALCULADOS — cacheada, nunca se recalcula ni se
# vuelve a descargar nada al recargar la página.
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando procesos...")
def cargar_procesos() -> pd.DataFrame:
    df = pd.read_parquet(RAIZ_TAREA2 / "data" / "processed" / "procesos_validados.parquet")
    df["fecha_publicacion"] = pd.to_datetime(df["fecha_publicacion"], errors="coerce", utc=True)
    return df


@st.cache_data
def cargar_geojson() -> dict:
    ruta = RAIZ_TAREA2 / "data" / "processed" / "departamentos.geojson"
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def cargar_reporte(nombre: str) -> pd.DataFrame | None:
    ruta = RAIZ_TAREA2 / "data" / "outputs" / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_resource
def obtener_config() -> dict:
    return cargar_config()


df_todo = cargar_procesos()
geojson = cargar_geojson()
config_base = obtener_config()

if df_todo.empty:
    st.error(
        "No hay procesos en `data/processed/procesos_validados.parquet`. "
        "Corre `src/adquisicion.py`, `src/procesar.py` y `src/validar.py` primero."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────
# Sidebar — filtros
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filtros")

deptos_presentes = sorted(d for d in df_todo["departamento_normalizado"].dropna().unique())
depto_sel = st.sidebar.multiselect("Departamento", deptos_presentes)

categorias_presentes = sorted(df_todo["categoria"].dropna().unique())
categoria_sel = st.sidebar.multiselect("Categoría", categorias_presentes)

monto_max_datos = float(df_todo["monto"].max())
rango_monto = st.sidebar.slider(
    "Rango de monto (PEN)", 0.0, monto_max_datos, (0.0, monto_max_datos), format="%.0f"
)

fechas_validas = df_todo["fecha_publicacion"].dropna()
if len(fechas_validas):
    fecha_min, fecha_max = fechas_validas.min().date(), fechas_validas.max().date()
    rango_fecha = st.sidebar.date_input("Rango de fechas", (fecha_min, fecha_max), min_value=fecha_min, max_value=fecha_max)
else:
    rango_fecha = None

umbral_similitud = st.sidebar.slider(
    "Umbral de similitud (solo afecta la pestaña Preguntar)",
    0.50, 1.00, float(config_base["recuperacion"]["umbral_similitud"]), 0.01,
)
st.sidebar.caption(f"Calibrado en Fase 3: {config_base['recuperacion']['umbral_similitud']}")

# ─────────────────────────────────────────────────────────────────────────
# Aplicar filtros — maneja selección vacía sin fallar
# ─────────────────────────────────────────────────────────────────────────
df = df_todo.copy()
if depto_sel:
    df = df[df["departamento_normalizado"].isin(depto_sel)]
if categoria_sel:
    df = df[df["categoria"].isin(categoria_sel)]
df = df[(df["monto"] >= rango_monto[0]) & (df["monto"] <= rango_monto[1])]
if isinstance(rango_fecha, tuple) and len(rango_fecha) == 2:
    inicio, fin = pd.Timestamp(rango_fecha[0], tz="UTC"), pd.Timestamp(rango_fecha[1], tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["fecha_publicacion"] >= inicio) & (df["fecha_publicacion"] < fin)]

# ─────────────────────────────────────────────────────────────────────────
# Encabezado KPI — se recalcula con los filtros de arriba
# ─────────────────────────────────────────────────────────────────────────
st.title("📡 Radar de Contratación Pública · Perú")
st.caption("Fuente: OECE (SEACE v3), 3 meses de 2026 (jun-ago) · Tarea 2")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Procesos", f"{len(df):,}")
c2.metric("Monto total (PEN)", f"{df['monto'].sum():,.0f}")
c3.metric("Departamentos representados", df["departamento_normalizado"].nunique())

resumen_riesgo = cargar_reporte("fase5_resumen.csv")
if resumen_riesgo is not None:
    proporcion_riesgo = resumen_riesgo.iloc[0]["proporcion_nacional_un_solo_licitador"] * 100
    c4.metric(
        "Adjudicado a un solo licitador",
        f"{proporcion_riesgo:.1f}%",
        help="Fase 5 — % de procesos ADJUDICADOS (a nivel nacional, 3 meses) en los que "
             "participó un único postor. Es una SEÑAL DE ALERTA que amerita revisión "
             "adicional, no una prueba de irregularidad — ver la pestaña 🚩 Riesgo.",
    )
else:
    c4.metric("Adjudicado a un solo licitador", "—", help="Corre `src/riesgo.py` primero.")

if df.empty:
    st.warning("Ningún proceso cumple con estos filtros. Ajusta los filtros de la izquierda.")

pestana_resumen, pestana_preguntar, pestana_tabla, pestana_riesgo, pestana_calidad = st.tabs(
    ["🗺️ Resumen", "💬 Preguntar", "📋 Tabla", "🚩 Riesgo", "✅ Calidad de datos"]
)

# ─────────────────────────────────────────────────────────────────────────
# Pestaña 1: mapa coroplético + distribución
# ─────────────────────────────────────────────────────────────────────────
with pestana_resumen:
    st.subheader("Mapa por departamento")
    if not geojson:
        st.info("No se encontró `data/processed/departamentos.geojson` — corre `src/preparar_geojson.py`.")
    elif df.empty:
        st.info("Sin datos para mostrar en el mapa con estos filtros.")
    else:
        color_por = st.radio("Colorear por", ["Número de procesos", "Monto total"], horizontal=True)
        agregado = (
            df.groupby("departamento_normalizado")
            .agg(n_procesos=("ocid", "count"), monto_total=("monto", "sum"))
            .reindex(DEPARTAMENTOS_OFICIALES, fill_value=0)
            .reset_index()
            .rename(columns={"index": "departamento_normalizado"})
        )
        columna_color = "n_procesos" if color_por == "Número de procesos" else "monto_total"
        # `px.choropleth` (el trazador "geo", basado en SVG) renderiza mal este
        # geojson: solo dibuja UN departamento y pinta el resto de la vista con
        # el color del último valor de la serie (bug reproducido y confirmado
        # con capturas reales, independiente de la versión de Plotly instalada,
        # 6.1.1 y 7.1.0). `px.choropleth_map` usa un renderizador distinto
        # (MapLibre, vectorial) que sí dibuja los 25 departamentos correctamente
        # — no necesita token, "carto-positron" es un estilo abierto.
        fig_mapa = px.choropleth_map(
            agregado,
            geojson=geojson,
            locations="departamento_normalizado",
            featureidkey="properties.departamento",
            color=columna_color,
            color_continuous_scale="Blues",
            hover_name="departamento_normalizado",
            hover_data={"n_procesos": True, "monto_total": ":,.0f", "departamento_normalizado": False},
            labels={"n_procesos": "Procesos", "monto_total": "Monto total (PEN)"},
            map_style="carto-positron",
            zoom=4.2,
            center={"lat": -9.5, "lon": -75.5},
            opacity=0.8,
        )
        fig_mapa.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=480)
        st.plotly_chart(fig_mapa, use_container_width=True)

    st.subheader("Distribución")
    col_izq, col_der = st.columns(2)
    if not df.empty:
        with col_izq:
            por_categoria = df.groupby("categoria").agg(n_procesos=("ocid", "count"), monto_total=("monto", "sum")).reset_index()
            fig_cat = px.bar(por_categoria, x="categoria", y="n_procesos", title="Procesos por categoría")
            st.plotly_chart(fig_cat, use_container_width=True)
        with col_der:
            df_mes = df.copy()
            df_mes["mes"] = df_mes["fecha_publicacion"].dt.strftime("%Y-%m")
            por_mes = df_mes.groupby("mes").agg(n_procesos=("ocid", "count")).reset_index().sort_values("mes")
            fig_mes = px.bar(por_mes, x="mes", y="n_procesos", title="Procesos por mes")
            st.plotly_chart(fig_mes, use_container_width=True)
    else:
        st.info("Sin datos para graficar con estos filtros.")

# ─────────────────────────────────────────────────────────────────────────
# Pestaña 2: preguntar (RAG híbrido de Fase 3)
# ─────────────────────────────────────────────────────────────────────────
with pestana_preguntar:
    st.info(
        f"Índice: {len(df_todo):,} procesos (jun-ago 2026). El filtro de departamento/categoría/monto "
        "de la izquierda NO se aplica aquí — la pregunta misma puede pedir un departamento/categoría/monto "
        "y el motor los detecta como filtros exactos (ver Fase 3)."
    )
    with st.form("formulario_pregunta_t2"):
        pregunta = st.text_area(
            "Tu pregunta",
            placeholder="Ej: Obras de agua y alcantarillado en Cusco por encima de un millón de soles",
            height=90,
        )
        enviar = st.form_submit_button("Preguntar")

    if enviar and pregunta.strip():
        config_consulta = {**config_base, "recuperacion": {**config_base["recuperacion"], "umbral_similitud": umbral_similitud}}
        with st.spinner("Filtrando, recuperando y generando la respuesta..."):
            try:
                resultado = responder(pregunta.strip(), config_consulta)
            except ErrorDeAPI as err:
                st.error(f"**Error del proveedor de generación** (`{err.proveedor}`): {err}")
                resultado = None

        if resultado is not None:
            if resultado["filtros_aplicados"]:
                st.caption(f"Filtros detectados en la pregunta: `{resultado['filtros_aplicados']}`")
            if resultado["abstuvo"]:
                st.warning("⚠️ El motor se abstuvo — no encontró procesos suficientemente relevantes.")
            else:
                st.success("Respuesta generada con los procesos recuperados.")

            st.markdown("### Respuesta")
            st.write(resultado["respuesta"])

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Costo", f"${resultado['costo_usd']:.6f}")
            mc2.metric("Mejor similitud", f"{resultado['mejor_similitud']:.4f}")
            mc3.metric("¿Se abstuvo?", "Sí" if resultado["abstuvo"] else "No")

            st.markdown("### Procesos recuperados")
            tabla_fuentes = pd.DataFrame(resultado["fuentes"])
            if not tabla_fuentes.empty:
                st.dataframe(
                    tabla_fuentes[["ocid", "titulo", "departamento", "categoria", "monto", "similitud"]],
                    width="stretch", hide_index=True,
                )
    elif enviar:
        st.error("Escribe una pregunta antes de enviar.")

# ─────────────────────────────────────────────────────────────────────────
# Pestaña 3: tabla ordenable + descarga CSV
# ─────────────────────────────────────────────────────────────────────────
with pestana_tabla:
    st.caption(f"{len(df):,} procesos con los filtros actuales — click en el encabezado de una columna para ordenar.")
    columnas_tabla = [
        "ocid", "titulo", "departamento_normalizado", "categoria", "monto",
        "comprador_nombre", "fecha_publicacion", "mes_origen",
    ]
    if df.empty:
        st.info("Sin filas con estos filtros.")
    else:
        st.dataframe(df[columnas_tabla], width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Descargar CSV (filtrado)",
            data=df[columnas_tabla].to_csv(index=False).encode("utf-8"),
            file_name="procesos_filtrados.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────
# Pestaña "Riesgo": indicador de Fase 5 — adjudicaciones a un solo licitador
# ─────────────────────────────────────────────────────────────────────────
with pestana_riesgo:
    st.warning(
        "⚠️ **Esta es una señal de alerta, no una prueba de irregularidad.** Una alta "
        "proporción de adjudicaciones a un solo licitador puede reflejar un mercado con "
        "pocos proveedores capaces, barreras de entrada legítimas, o competencia "
        "restringida indebidamente — distinguir entre esas causas requiere investigación "
        "adicional, no solo este número. No se publican nombres de personas, solo "
        "entidades compradoras (municipalidades, ministerios, etc.).\n\n"
        "Referencias: [Open Contracting Partnership — Red Flags in Public Procurement "
        "(2024)](https://www.open-contracting.org/resources/red-flags-in-public-procurement-a-guide-to-using-data-to-detect-and-mitigate-risks/) "
        "· [Ojo Público — Funes, un algoritmo contra la corrupción](https://ojo-publico.com/especiales/funes/)"
    )

    st.subheader("Metodología")
    st.markdown(
        "- **Adjudicado**: el proceso tiene al menos una liberación en `awards` con "
        "proveedor asignado (OCDS no trae `tender.status`/`award.status` poblados en "
        "este dataset).\n"
        "- **Un solo licitador**: `tender.numberOfTenderers == 1` entre los adjudicados "
        "— cuántos postores PARTICIPARON, no cuántos ganaron.\n"
        f"- **Mínimo de procesos por comprador**: {config_base.get('riesgo', {}).get('umbral_minimo_procesos_por_comprador', 10)} "
        "adjudicados — con menos, un solo caso mueve la proporción en ≥10 puntos "
        "porcentuales (ruido, no patrón), así que se excluye del ranking."
    )

    resumen_r = cargar_reporte("fase5_resumen.csv")
    if resumen_r is not None:
        r = resumen_r.iloc[0]
        mr1, mr2, mr3 = st.columns(3)
        mr1.metric("Procesos adjudicados", f"{int(r['n_adjudicados']):,}")
        mr2.metric("Con un solo licitador", f"{int(r['n_un_solo_licitador']):,}")
        mr3.metric(
            "Compradores en el ranking",
            f"{int(r['n_compradores_con_minimo'])} / {int(r['n_compradores_evaluados_total'])}",
            help="Con al menos el mínimo de procesos adjudicados definido arriba.",
        )
    else:
        st.info("Todavía no existe `data/outputs/fase5_resumen.csv` — corre `src/riesgo.py`.")

    st.subheader("Por departamento")
    tabla_depto_riesgo = cargar_reporte("fase5_riesgo_departamento.csv")
    if tabla_depto_riesgo is not None:
        tabla_depto_riesgo = tabla_depto_riesgo.copy()
        tabla_depto_riesgo["proporcion_un_solo_licitador"] = (
            tabla_depto_riesgo["proporcion_un_solo_licitador"] * 100
        ).round(1)
        st.dataframe(
            tabla_depto_riesgo.rename(columns={
                "departamento": "Departamento", "n_adjudicados": "Adjudicados",
                "n_un_solo_licitador": "Con un solo licitador",
                "proporcion_un_solo_licitador": "Proporción (%)",
            }),
            width="stretch", hide_index=True,
        )
    else:
        st.caption("Todavía no existe `data/outputs/fase5_riesgo_departamento.csv`.")

    st.subheader(f"Top 10 compradores (mínimo {config_base.get('riesgo', {}).get('umbral_minimo_procesos_por_comprador', 10)} procesos adjudicados)")
    tabla_compradores_riesgo = cargar_reporte("fase5_riesgo_top_compradores.csv")
    if tabla_compradores_riesgo is not None:
        tabla_compradores_riesgo = tabla_compradores_riesgo.copy()
        tabla_compradores_riesgo["proporcion_un_solo_licitador"] = (
            tabla_compradores_riesgo["proporcion_un_solo_licitador"] * 100
        ).round(1)
        st.dataframe(
            tabla_compradores_riesgo.rename(columns={
                "comprador_nombre": "Entidad compradora", "n_adjudicados": "Adjudicados",
                "n_un_solo_licitador": "Con un solo licitador",
                "proporcion_un_solo_licitador": "Proporción (%)",
            }),
            width="stretch", hide_index=True,
        )
    else:
        st.caption("Todavía no existe `data/outputs/fase5_riesgo_top_compradores.csv`.")

# ─────────────────────────────────────────────────────────────────────────
# Pestaña 4: calidad de datos (Fase 1 + Fase 2)
# ─────────────────────────────────────────────────────────────────────────
with pestana_calidad:
    st.subheader("Fase 1 — Calidad de extracción/adquisición")
    tabla_f1 = cargar_reporte("fase1_calidad.csv")
    if tabla_f1 is not None:
        st.dataframe(tabla_f1, width="stretch", hide_index=True)
    else:
        st.caption("Todavía no existe `data/outputs/fase1_calidad.csv`.")

    st.subheader("Fase 2 — Validación y normalización territorial")
    tabla_f2 = cargar_reporte("fase2_calidad_validacion.csv")
    if tabla_f2 is not None:
        st.dataframe(tabla_f2, width="stretch", hide_index=True)
        st.caption(
            "Cada fila es una regla de calidad: cuántos registros marcó y qué se hizo con ellos "
            "(corregidos, eliminados con justificación, o conservados con advertencia)."
        )
    else:
        st.caption("Todavía no existe `data/outputs/fase2_calidad_validacion.csv`.")

    st.subheader("Fase 3 — Recall@k del motor RAG híbrido")
    tabla_f3 = cargar_reporte("fase3_resumen.csv")
    if tabla_f3 is not None:
        st.dataframe(tabla_f3, width="stretch", hide_index=True)
    else:
        st.caption("Todavía no existe `data/outputs/fase3_resumen.csv`.")

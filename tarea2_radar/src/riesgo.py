r"""Fase 5 de la Tarea 2 — indicador de riesgo "adjudicaciones a un solo
licitador".

## Qué mide y por qué es una señal de alerta, no una prueba

Cuando un proceso de contratación se adjudica y solo UN postor participó en
la competencia (ni siquiera que haya ganado un solo postor — que PARTICIPÓ
uno solo), no hubo competencia real de precios ni de condiciones. La
literatura de integridad en contrataciones públicas lo trata como una señal
de alerta clásica:

- Open Contracting Partnership, "Red Flags in Public Procurement" (2024):
  cataloga ~73 indicadores de alerta a partir de datos OCDS; el número de
  postores por proceso es uno de los campos base que alimenta varios de
  esos indicadores de riesgo de manipulación de la competencia.
- Ojo Público, "Funes, un algoritmo contra la corrupción": usa el patrón de
  "postor único" repetido en un mismo proveedor/comprador como una de las
  señales que activan revisión manual — no como veredicto. Un caso citado:
  una empresa fue postora única en 90% de sus contratos, lo cual disparó
  investigación periodística, no una acusación directa.

Ambas fuentes coinciden en el mismo punto, que replicamos aquí de forma
explícita en el panel y se debe repetir en el video: **una proporción alta
de adjudicaciones a un solo licitador es una señal de que hay que investigar
más a fondo — mercado con pocos proveedores capaces, barreras de entrada
legítimas, o competencia restringida indebidamente son todas explicaciones
posibles. No es, por sí sola, prueba de irregularidad.** Por eso este
módulo NUNCA imprime nombres de personas: `comprador_nombre` en OCDS es
siempre una ENTIDAD (municipalidad, ministerio, hospital...), nunca un
funcionario individual — verificado en las 2,183 entidades del corpus (todas
en mayúscula, formato institucional), y filtrado por seguridad con
`_filtrar_nombres_no_institucionales()` antes de publicar el ranking.

## Definiciones (con los campos OCDS realmente disponibles)

- **Adjudicado**: el proceso tiene al menos una liberación en `awards` con
  un proveedor (`suppliers`) asignado. OCDS trae `tender.status` y
  `award.status` vacíos en el 100% de los 20,424 procesos de este dataset
  (verificado), así que no se puede filtrar por esos campos; la presencia
  de `awards` es la señal disponible de que el proceso llegó a buena pro.
- **Número de licitadores**: `tender.numberOfTenderers` — cuántos postores
  PARTICIPARON en la competencia, no cuántos ganaron (eso sería
  `len(awards[].suppliers)`, que es casi siempre 1 y no mide competencia).
- **Un solo licitador**: `numero_licitadores == 1` entre los adjudicados.

## Número mínimo de procesos (justificación)

Con muy pocos procesos, la proporción es inestable: un comprador con 1 solo
proceso adjudicado y ese proceso con un único postor tiene "100% de riesgo"
sin que eso signifique nada — es un solo caso, no un patrón. Con n=5,
un solo proceso mueve la proporción en 20 puntos porcentuales; con n=10, la
mueve en 10 puntos. Se fija el mínimo en **10 procesos adjudicados** por
comprador (y por departamento, aunque en la práctica todos los
departamentos superan ese umbral con margen) — por debajo de eso, un único
proceso puede seguir dominando la lectura, así que se excluye para no
convertir ruido estadístico en un ranking de "los peores compradores".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIR_PROCESSED = RAIZ_PROYECTO / "data" / "processed"
DIR_REPORTS = RAIZ_PROYECTO / "data" / "outputs"


def _cargar_config_riesgo() -> dict:
    with open(RAIZ_PROYECTO / "config.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config["riesgo"]


def _es_nombre_institucional(nombre: str) -> bool:
    """Heurística real (no NER completo, pero no un no-op): las 2,183
    entidades compradoras del corpus están, sin excepción, en MAYÚSCULA
    (verificado). Un nombre de persona pegado por error en este campo
    aparecería casi seguro en mayúscula/minúscula mixta ('Juan Pérez'), así
    que se usa eso como filtro de seguridad antes de publicar cualquier
    ranking por comprador."""
    return nombre == nombre.upper()


def _filtrar_nombres_no_institucionales(tabla: pd.DataFrame, columna: str) -> pd.DataFrame:
    es_institucional = tabla[columna].astype(str).apply(_es_nombre_institucional)
    if (~es_institucional).any():
        excluidos = tabla.loc[~es_institucional, columna].tolist()
        print(f"AVISO Fase 5: {len(excluidos)} nombre(s) de comprador no parecen institucionales "
              f"(no están en mayúscula) — excluidos del ranking por precaución: {excluidos}")
    return tabla.loc[es_institucional].copy()


def calcular() -> dict:
    config_riesgo = _cargar_config_riesgo()
    umbral_minimo_procesos = config_riesgo["umbral_minimo_procesos_por_comprador"]
    top_n_compradores = config_riesgo["top_n_compradores"]

    df = pd.read_parquet(DIR_PROCESSED / "procesos_validados.parquet")

    adjudicados = df[df["adjudicado"] & df["numero_licitadores"].notna()].copy()
    adjudicados["un_solo_licitador"] = adjudicados["numero_licitadores"] == 1

    proporcion_nacional = adjudicados["un_solo_licitador"].mean()

    # --- Por departamento ---
    por_departamento = (
        adjudicados.groupby("departamento_normalizado")
        .agg(n_adjudicados=("ocid", "count"), n_un_solo_licitador=("un_solo_licitador", "sum"))
        .reset_index()
        .rename(columns={"departamento_normalizado": "departamento"})
    )
    por_departamento["proporcion_un_solo_licitador"] = (
        por_departamento["n_un_solo_licitador"] / por_departamento["n_adjudicados"]
    )
    por_departamento = por_departamento.sort_values("proporcion_un_solo_licitador", ascending=False)

    # --- Por comprador (entidad), con el mínimo de procesos ---
    por_comprador = (
        adjudicados.groupby("comprador_nombre")
        .agg(n_adjudicados=("ocid", "count"), n_un_solo_licitador=("un_solo_licitador", "sum"))
        .reset_index()
    )
    por_comprador["proporcion_un_solo_licitador"] = (
        por_comprador["n_un_solo_licitador"] / por_comprador["n_adjudicados"]
    )
    por_comprador = _filtrar_nombres_no_institucionales(por_comprador, "comprador_nombre")
    compradores_con_minimo = por_comprador[por_comprador["n_adjudicados"] >= umbral_minimo_procesos].copy()
    top_compradores = compradores_con_minimo.sort_values(
        ["proporcion_un_solo_licitador", "n_adjudicados"], ascending=[False, False]
    ).head(top_n_compradores)

    DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    por_departamento.to_csv(DIR_REPORTS / "fase5_riesgo_departamento.csv", index=False)
    top_compradores.to_csv(DIR_REPORTS / "fase5_riesgo_top_compradores.csv", index=False)

    resumen = {
        "n_procesos_total": len(df),
        "n_adjudicados": len(adjudicados),
        "n_un_solo_licitador": int(adjudicados["un_solo_licitador"].sum()),
        "proporcion_nacional_un_solo_licitador": round(float(proporcion_nacional), 4),
        "umbral_minimo_procesos_por_comprador": umbral_minimo_procesos,
        "n_compradores_evaluados_total": len(por_comprador),
        "n_compradores_con_minimo": len(compradores_con_minimo),
        "n_compradores_excluidos_por_pocos_procesos": len(por_comprador) - len(compradores_con_minimo),
    }
    pd.DataFrame([resumen]).to_csv(DIR_REPORTS / "fase5_resumen.csv", index=False)

    return resumen


if __name__ == "__main__":
    resumen = calcular()
    print("=== Fase 5 — indicador de riesgo: adjudicaciones a un solo licitador ===")
    for clave, valor in resumen.items():
        print(f"{clave}: {valor}")
    print(
        "\nRECORDATORIO: esta es una señal de alerta que amerita revisión adicional, "
        "NO una prueba de irregularidad. No se publican nombres de personas, solo "
        "entidades compradoras."
    )

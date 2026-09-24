r"""Fase 2 de la Tarea 2 — validación y normalización territorial.

Corre sobre `data/processed/procesos.parquet` (salida de `procesar.py`).
Cada regla se aplica de forma explícita, se cuenta cuántas filas afecta, y
se registra qué se hizo con ellas — nunca se elimina nada en silencio.

## Los 25 departamentos del Perú y la reconciliación de acentos

Perú tiene 24 departamentos + la Provincia Constitucional del Callao = 25
unidades de nivel regional. Los NOMBRES OFICIALES llevan tilde (INEI):
"JUNÍN", "HUÁNUCO", "SAN MARTÍN", "APURÍMAC"... pero el campo `department`
que exporta la API de OECE viene en mayúsculas SIN tilde ("JUNIN",
"HUANUCO"...) — verificado en los 20,424 procesos reales: ninguno de los 25
valores únicos de `departamento` lleva tilde. Esa es exactamente la
inconsistencia "JUNÍN vs. JUNIN" del enunciado: no aparece como dos grafías
distintas DENTRO del dataset de OECE (que es internamente consistente, sin
tilde), sino entre el dataset de OECE y el nombre OFICIAL que se necesita
para casar con otras fuentes (ej. los polígonos departamentales, que sí
suelen traer tilde). La regla de normalización (`normalizar_departamento`)
resuelve esto quitando tildes y pasando a mayúsculas ANTES de comparar,
para que "JUNÍN", "Junín", "junin" y "JUNIN" casen con el mismo departamento
canónico — y se prueba explícitamente con esos 4 casos, no solo con los
datos ya limpios que nos tocaron.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIR_PROCESSED = RAIZ_PROYECTO / "data" / "processed"
DIR_REPORTS = RAIZ_PROYECTO / "data" / "outputs"

# Nombres oficiales (INEI), con tilde donde corresponde.
DEPARTAMENTOS_OFICIALES = [
    "AMAZONAS", "ANCASH", "APURÍMAC", "AREQUIPA", "AYACUCHO", "CAJAMARCA",
    "CALLAO", "CUSCO", "HUANCAVELICA", "HUÁNUCO", "ICA", "JUNÍN",
    "LA LIBERTAD", "LAMBAYEQUE", "LIMA", "LORETO", "MADRE DE DIOS",
    "MOQUEGUA", "PASCO", "PIURA", "PUNO", "SAN MARTÍN", "TACNA", "TUMBES",
    "UCAYALI",
]
assert len(DEPARTAMENTOS_OFICIALES) == 25


def _sin_tildes(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def _clave_normalizada(texto: str) -> str:
    return _sin_tildes(str(texto)).strip().upper()


_MAPA_DEPARTAMENTOS = {_clave_normalizada(d): d for d in DEPARTAMENTOS_OFICIALES}


def normalizar_departamento(valor) -> str | None:
    """Regla explícita: quitar tildes, pasar a mayúsculas, quitar espacios
    sobrantes, y buscar en la lista oficial de 25 departamentos. Devuelve
    el nombre oficial (con tilde) si hay match, None si no se pudo ubicar."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    return _MAPA_DEPARTAMENTOS.get(_clave_normalizada(valor))


def _autotest_normalizacion() -> None:
    """Prueba la regla con los casos del enunciado, no solo con datos ya
    limpios — para demostrar que funciona, no solo que no rompió nada."""
    casos = {
        "JUNIN": "JUNÍN", "JUNÍN": "JUNÍN", "Junín": "JUNÍN", "junin": "JUNÍN",
        "  LIMA  ": "LIMA", "callao": "CALLAO", "San Martin": "SAN MARTÍN",
        "PROVINCIA_INEXISTENTE": None,
    }
    for entrada, esperado in casos.items():
        obtenido = normalizar_departamento(entrada)
        assert obtenido == esperado, f"'{entrada}' -> '{obtenido}', se esperaba '{esperado}'"
    print(f"Autotest de normalizar_departamento: {len(casos)}/{len(casos)} casos OK "
          "(incluye acentos, mayúsculas, espacios y un valor inexistente)")


def validar() -> dict:
    _autotest_normalizacion()

    df = pd.read_parquet(DIR_PROCESSED / "procesos.parquet")
    n_inicial = len(df)
    reglas = []

    # 1) Registros repetidos para el mismo proceso (mismo ocid)
    duplicados = df["ocid"].duplicated(keep="first")
    n_duplicados = int(duplicados.sum())
    if n_duplicados:
        df = df.loc[~duplicados].copy()
    reglas.append({
        "regla": "ocid_duplicado",
        "descripcion": "Más de una fila para el mismo proceso de contratación (mismo ocid)",
        "filas_marcadas": n_duplicados,
        "accion": "eliminadas (se conserva la primera)" if n_duplicados else "ninguna — 0 encontrados",
        "nota": "Fase 1 (procesar.py) ya deduplica por ocid entre meses; esta regla re-verifica "
                "de forma independiente sobre la salida, sin confiar ciegamente en ello.",
    })

    # 2) Monto faltante o cero
    monto_nulo = df["monto"].isna()
    monto_cero = df["monto"].fillna(-1) == 0
    monto_problema = monto_nulo | monto_cero
    df["advertencia_monto_cero_o_faltante"] = monto_problema
    reglas.append({
        "regla": "monto_faltante_o_cero",
        "descripcion": "El proceso no tiene monto, o el monto es 0",
        "filas_marcadas": int(monto_problema.sum()),
        "accion": "conservadas con advertencia (columna advertencia_monto_cero_o_faltante) "
                  "— un monto en 0 puede ser legítimo (ej. acuerdos marco sin valor fijado en "
                  "la convocatoria); eliminar ~13% del corpus sin evidencia de que sea un error "
                  "sería una pérdida de datos injustificada",
        "nota": f"{int(monto_nulo.sum())} sin monto, {int(monto_cero.sum())} con monto=0",
    })

    # 3) Procesos sin descripción
    desc_vacia = df["descripcion"].isna() | df["descripcion"].astype(str).str.strip().eq("")
    reglas.append({
        "regla": "sin_descripcion",
        "descripcion": "El proceso no tiene descripción (ni de tender.description)",
        "filas_marcadas": int(desc_vacia.sum()),
        "accion": "ninguna — 0 encontrados en este corpus" if desc_vacia.sum() == 0
        else "eliminadas (sin descripción no hay nada que indexar semánticamente)",
        "nota": "",
    })
    if desc_vacia.any():
        df = df.loc[~desc_vacia].copy()

    # 4) Campo de ubicación cuyos valores no son departamentos (found in
    #    the OCDS 'region' field, not used downstream — documented here).
    valores_region = df["region"].dropna().astype(str).unique()
    region_es_departamento = {v: (_clave_normalizada(v) in _MAPA_DEPARTAMENTOS) for v in valores_region}
    n_region_no_depto = sum(not es for es in region_es_departamento.values())
    reglas.append({
        "regla": "ubicacion_no_es_departamento",
        "descripcion": "El campo OCDS 'region' de la dirección del comprador NO contiene "
                        "departamentos, contiene provincias (verificado contra la lista oficial "
                        "de 25 departamentos)",
        "filas_marcadas": int(df["region"].notna().sum()),
        "accion": "campo 'region' IGNORADO para la columna de territorio; se usa 'departamento' "
                   "(campo OCDS distinto, de la extensión ocds_department_extension de OECE, "
                   "que sí trae el departamento correcto) en su lugar",
        "nota": f"{n_region_no_depto}/{len(valores_region)} valores únicos de 'region' NO son "
                "nombres de departamento (son provincias, ej. 'HUANCAYO', 'TRUJILLO', 'CHICLAYO')",
    })

    # 5) Normalización territorial a los 25 departamentos + reconciliación
    #    de acentos (JUNÍN vs JUNIN)
    df["departamento_normalizado"] = df["departamento"].apply(normalizar_departamento)
    no_localizados = df["departamento_normalizado"].isna()
    reglas.append({
        "regla": "normalizacion_25_departamentos",
        "descripcion": "Normalizar 'departamento' (sin tilde, OECE) al nombre oficial INEI "
                        "(con tilde) de los 25 departamentos, quitando tildes/mayúsculas/espacios "
                        "antes de comparar — así 'JUNIN' y 'JUNÍN' casan con el mismo departamento",
        "filas_marcadas": int(no_localizados.sum()),
        "accion": "corregidas (mapeadas al nombre oficial)" if not no_localizados.all()
        else "0 sin ubicar",
        "nota": f"{int(no_localizados.sum())} procesos no se pudieron ubicar en ninguno de los 25 "
                f"departamentos ({round(no_localizados.mean() * 100, 2)}%)"
                + (f" — motivo: valor de 'departamento' = {sorted(df.loc[no_localizados, 'departamento'].unique())}"
                   if no_localizados.any() else ""),
    })

    n_final = len(df)

    tabla_reglas = pd.DataFrame(reglas)
    DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    tabla_reglas.to_csv(DIR_REPORTS / "fase2_calidad_validacion.csv", index=False)

    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DIR_PROCESSED / "procesos_validados.parquet", index=False)
    df.to_csv(DIR_PROCESSED / "procesos_validados.csv", index=False)

    return {
        "n_filas_inicial": n_inicial,
        "n_filas_final": n_final,
        "n_departamentos_normalizados": int((~no_localizados).sum()),
        "n_no_localizados": int(no_localizados.sum()),
        "reglas": reglas,
    }


if __name__ == "__main__":
    resumen = validar()
    print(f"\nFilas iniciales: {resumen['n_filas_inicial']}")
    print(f"Filas finales:   {resumen['n_filas_final']}")
    print(f"Departamentos normalizados: {resumen['n_departamentos_normalizados']}")
    print(f"No localizados: {resumen['n_no_localizados']}")
    print("\n=== Reglas ===")
    for r in resumen["reglas"]:
        print(f"- {r['regla']}: {r['filas_marcadas']} marcadas -> {r['accion']}")
        if r["nota"]:
            print(f"    nota: {r['nota']}")

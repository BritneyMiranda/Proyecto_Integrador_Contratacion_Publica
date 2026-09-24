r"""Fase 1 de la Tarea 2 — de los ZIP mensuales de OECE a UNA fila por
proceso de contratación.

## Modelo de datos OCDS: release vs. record

Cada archivo mensual que descarga `adquisicion.py` es un **"record
package"** de OCDS (clave de nivel superior `"records"`, no `"releases"`).
Cada elemento de `records[]` tiene tres campos:

    {"ocid": "...", "releases": [...], "compiledRelease": {...}}

- Una **liberación (release)** es un documento OCDS que captura UN evento o
  etapa puntual de un proceso de contratación (ej. la convocatoria, una
  modificación de bases, la buena pro) tal como estaba en ese momento. Cada
  liberación tiene su propio `id` (que en este dataset se arma como
  `ocid + timestamp de compilación`), pero varias liberaciones pueden
  compartir el mismo proceso.
- Lo que identifica una liberación como perteneciente a un mismo proceso es
  el **`ocid`** (Open Contracting ID) — ese es el campo que agrupa todas las
  liberaciones de un mismo proceso de contratación a lo largo del tiempo.
- Un **registro (record)** es el contenedor de OCDS para UN `ocid`: guarda
  la lista completa de liberaciones (`releases`) MÁS una liberación
  fusionada (`compiledRelease`) que resume el estado más reciente de todo
  el proceso. Por diseño, hay como máximo un `compiledRelease` por `ocid`.

Por eso este script arma la tabla final a partir de `compiledRelease`, NO de
`releases`: eso es justamente lo que garantiza una fila por proceso DENTRO
de un mismo archivo mensual. El problema real, verificado con los 3 meses
descargados, es que el MISMO `ocid` puede aparecer en más de un archivo
mensual (un proceso publicado en junio puede seguir activo y reaparecer,
actualizado, en el archivo de julio) — así que además del compilado por
mes, este script deduplica por `ocid` ACROSS los 3 meses, quedándose con la
versión de fecha de compilación (`compiledRelease.date`) más reciente.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ_PROYECTO / "data" / "raw"
DIR_PROCESSED = RAIZ_PROYECTO / "data" / "processed"
DIR_REPORTS = RAIZ_PROYECTO / "data" / "outputs"


def _leer_zip_mensual(ruta_zip: Path) -> dict:
    with zipfile.ZipFile(ruta_zip) as z:
        nombre_interno = z.namelist()[0]
        with z.open(nombre_interno) as fh:
            return json.load(fh)


def _direccion_comprador(compiled: dict) -> dict:
    """Busca, entre las `parties`, la que tiene rol 'buyer' y devuelve su
    dirección (departamento/región) — ahí vive la ubicación geográfica del
    proceso, como METADATA estructurada, no como texto libre."""
    comprador_id = (compiled.get("buyer") or {}).get("id")
    for party in compiled.get("parties", []):
        if party.get("id") == comprador_id or "buyer" in party.get("roles", []):
            return party.get("address", {}) or {}
    return {}


def _fila_desde_compiled(compiled: dict, mes_origen: str) -> dict:
    tender = compiled.get("tender", {}) or {}
    valor = tender.get("value", {}) or {}
    direccion = _direccion_comprador(compiled)
    buyer = compiled.get("buyer", {}) or {}

    return {
        "ocid": compiled.get("ocid"),
        "mes_origen": mes_origen,
        "compiled_date": compiled.get("date"),
        "tender_id": tender.get("id"),
        "titulo": tender.get("title"),
        "descripcion": tender.get("description"),
        "categoria": tender.get("mainProcurementCategory"),
        "metodo_contratacion": tender.get("procurementMethodDetails"),
        "monto": (valor.get("amount_PEN") if valor.get("amount_PEN") is not None else valor.get("amount")),
        "moneda": valor.get("currency"),
        "fecha_publicacion": tender.get("datePublished"),
        "comprador_id": buyer.get("id"),
        "comprador_nombre": buyer.get("name"),
        "departamento": direccion.get("department"),
        "region": direccion.get("region"),
        "localidad": direccion.get("locality"),
        "n_items": len(tender.get("items", [])),
        # Fase 5 — indicador de riesgo "adjudicaciones a un solo licitador":
        # 'adjudicado' = tiene al menos una liberación en 'awards' (OCDS no
        # trae tender.status/award.status poblados en este dataset, así que
        # la presencia de 'awards' es la señal disponible de que el proceso
        # llegó a buena pro). 'numero_licitadores' = tender.numberOfTenderers
        # (cuántos postores PARTICIPARON, no cuántos ganaron).
        "adjudicado": bool(compiled.get("awards")),
        "numero_licitadores": tender.get("numberOfTenderers"),
    }


def procesar() -> dict:
    filas = []
    n_records_por_mes = {}
    n_releases_por_record: list[int] = []

    for ruta_zip in sorted(DIR_RAW.glob("*.zip")):
        mes_origen = ruta_zip.stem.split("_")[0]  # "2026-06_seace_v3_json" -> "2026-06"
        paquete = _leer_zip_mensual(ruta_zip)
        records = paquete["records"]
        n_records_por_mes[mes_origen] = len(records)

        for record in records:
            compiled = record.get("compiledRelease")
            if not compiled or not compiled.get("ocid"):
                continue  # registro sin liberación fusionada: no se puede usar
            filas.append(_fila_desde_compiled(compiled, mes_origen))
            n_releases_por_record.append(len(record.get("releases", [])))

    tabla_antes = pd.DataFrame(filas)
    filas_antes = len(tabla_antes)

    # Deduplicar por ocid ACROSS los 3 meses: nos quedamos con la fila de
    # compiled_date más reciente para cada ocid.
    tabla_antes["compiled_date"] = pd.to_datetime(tabla_antes["compiled_date"], errors="coerce", utc=True)
    tabla_final = (
        tabla_antes.sort_values("compiled_date")
        .drop_duplicates(subset="ocid", keep="last")
        .reset_index(drop=True)
    )
    filas_despues = len(tabla_final)

    assert tabla_final["ocid"].is_unique, "quedaron ocid duplicados — no debería pasar"

    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    tabla_final.to_parquet(DIR_PROCESSED / "procesos.parquet", index=False)
    tabla_final.to_csv(DIR_PROCESSED / "procesos.csv", index=False)

    resumen = {
        "n_records_por_mes": n_records_por_mes,
        "filas_antes_de_deduplicar": filas_antes,
        "filas_despues_de_deduplicar": filas_despues,
        "ocid_duplicados_entre_meses": filas_antes - filas_despues,
        "releases_por_record_min": min(n_releases_por_record) if n_releases_por_record else None,
        "releases_por_record_mediana": pd.Series(n_releases_por_record).median() if n_releases_por_record else None,
        "releases_por_record_max": max(n_releases_por_record) if n_releases_por_record else None,
        "n_departamentos_distintos": tabla_final["departamento"].nunique(),
        "n_sin_departamento": int(tabla_final["departamento"].isna().sum()),
    }

    DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([resumen]).to_csv(DIR_REPORTS / "fase1_calidad.csv", index=False)

    return resumen


if __name__ == "__main__":
    resumen = procesar()
    print("=== Resumen Fase 1 (procesamiento) ===")
    for clave, valor in resumen.items():
        print(f"{clave}: {valor}")

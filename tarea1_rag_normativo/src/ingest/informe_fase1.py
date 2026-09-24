"""Orquesta la Fase 1 para todos los documentos declarados en la sección
`documentos:` de config.yaml: verifica, limpia, indexa a JSONL y arma el
informe de calidad de extracción."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .pipeline import MetadatosDocumento, procesar_documento

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]


def cargar_documentos(ruta_yaml: str | Path = RAIZ_PROYECTO / "config.yaml") -> list[dict]:
    with open(ruta_yaml, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config["documentos"] + config.get("opcionales", [])


def generar_informe_fase1(
    ruta_yaml: str | Path = RAIZ_PROYECTO / "config.yaml",
    dir_salida_processed: str | Path = RAIZ_PROYECTO / "data" / "processed",
    dir_reportes: str | Path = RAIZ_PROYECTO / "data" / "outputs",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Devuelve (tabla_resumen, ejemplos_antes_despues).

    Documentos cuyo PDF todavía no existe en data/raw se listan en la
    tabla con estado 'PDF no encontrado' en vez de fallar todo el informe.
    """
    documentos = cargar_documentos(ruta_yaml)
    filas = []
    ejemplos: dict[str, str] = {}

    for doc in documentos:
        ruta_pdf = RAIZ_PROYECTO / doc["ruta_pdf"]
        if not ruta_pdf.exists():
            filas.append(
                {
                    "doc_id": doc["doc_id"],
                    "paginas": None,
                    "caracteres_totales": None,
                    "caracteres_prom_pagina": None,
                    "paginas_sin_texto": None,
                    "orden_lectura_mediana": None,
                    "usable": "PDF no encontrado",
                }
            )
            continue

        meta = MetadatosDocumento(
            doc_id=doc["doc_id"],
            titulo=doc["titulo"],
            institucion=doc["institucion"],
            tipo=doc["tipo"],
            fuente_url=doc["fuente_url"],
            fecha_descarga=doc.get("fecha_descarga") or "sin_registrar",
            version=doc.get("version") or "sin_registrar",
        )

        reporte, ruta_jsonl = procesar_documento(ruta_pdf, meta, dir_salida_processed)
        filas.append(reporte.resumen())

        lineas = ruta_jsonl.read_text(encoding="utf-8").splitlines()
        if lineas:
            import json

            idx_medio = len(lineas) // 2
            pagina_media = json.loads(lineas[idx_medio])
            ejemplos[doc["doc_id"]] = pagina_media["texto"]

    tabla = pd.DataFrame(filas)

    dir_reportes = Path(dir_reportes)
    dir_reportes.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(dir_reportes / "fase1_calidad.csv", index=False)
    (dir_reportes / "fase1_calidad.md").write_text(tabla.to_markdown(index=False), encoding="utf-8")

    return tabla, ejemplos

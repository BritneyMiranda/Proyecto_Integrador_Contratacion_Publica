"""Pipeline de ingesta Fase 1: PDF crudo -> texto limpio por página en JSONL.

Este módulo es parte del "proceso fuera de línea": se corre una vez (o
cuando cambian los documentos fuente) y nunca lo toca el proceso que
responde preguntas en línea.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .clean import detectar_lineas_repetidas, limpiar_pagina
from .pdf_quality import ReporteCalidad, verificar_pdf


@dataclass
class MetadatosDocumento:
    doc_id: str
    titulo: str
    institucion: str
    tipo: str  # "ley" | "decreto_supremo" | ...
    fuente_url: str
    fecha_descarga: str  # YYYY-MM-DD
    version: str = "sin_registrar"  # fecha de promulgación/publicación de la norma


def procesar_documento(
    ruta_pdf: str | Path,
    meta: MetadatosDocumento,
    dir_salida: str | Path,
    min_fraction_boilerplate: float = 0.7,
) -> tuple[ReporteCalidad, Path]:
    """Verifica, limpia e indexa un PDF por página.

    Devuelve el reporte de calidad y la ruta del .jsonl generado.
    Lanza AssertionError si el documento no pasa la verificación de
    Fase 1 (ver `ReporteCalidad.usable`) — no tiene sentido limpiar un
    PDF que no se puede leer.
    """
    ruta_pdf = Path(ruta_pdf)
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    reporte = verificar_pdf(ruta_pdf, doc_id=meta.doc_id)
    if not reporte.usable:
        raise AssertionError(
            f"'{meta.doc_id}' no pasó la verificación de calidad "
            f"(ver reporte.resumen()): {reporte.resumen()}. "
            "Revisa el PDF antes de continuar — no se genera el JSONL."
        )

    reader = PdfReader(str(ruta_pdf))
    textos_crudos = [page.extract_text() or "" for page in reader.pages]

    boilerplate = detectar_lineas_repetidas(textos_crudos, min_fraction=min_fraction_boilerplate)

    ruta_salida = dir_salida / f"{meta.doc_id}.jsonl"
    with ruta_salida.open("w", encoding="utf-8") as fh:
        for i, crudo in enumerate(textos_crudos, start=1):
            limpio = limpiar_pagina(crudo, boilerplate)
            registro = {
                "doc_id": meta.doc_id,
                "titulo": meta.titulo,
                "institucion": meta.institucion,
                "tipo": meta.tipo,
                "fuente_url": meta.fuente_url,
                "fecha_descarga": meta.fecha_descarga,
                "version": meta.version,
                "pagina": i,
                "caracteres_crudos": len(crudo),
                "caracteres_limpios": len(limpio),
                "texto": limpio,
            }
            fh.write(json.dumps(registro, ensure_ascii=False) + "\n")

    return reporte, ruta_salida

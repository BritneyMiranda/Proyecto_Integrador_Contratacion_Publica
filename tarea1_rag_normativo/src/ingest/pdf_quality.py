"""Verificación de la calidad de extracción de un PDF.

Antes de construir cualquier índice, hay que confirmar que el texto se puede
extraer completo, en el orden correcto y sin páginas vacías. Este módulo no
limpia nada: solo mide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

MIN_CHARS_PARA_CONSIDERAR_CON_TEXTO = 20


@dataclass
class ReportePagina:
    pagina: int
    caracteres: int
    tiene_texto: bool
    orden_lectura_ratio: float | None  # None si la página no tiene palabras


@dataclass
class ReporteCalidad:
    doc_id: str
    ruta: str
    n_paginas: int
    paginas: list[ReportePagina] = field(default_factory=list)

    @property
    def total_caracteres(self) -> int:
        return sum(p.caracteres for p in self.paginas)

    @property
    def promedio_caracteres_pagina(self) -> float:
        return self.total_caracteres / self.n_paginas if self.n_paginas else 0.0

    @property
    def paginas_sin_texto(self) -> list[int]:
        return [p.pagina for p in self.paginas if not p.tiene_texto]

    @property
    def orden_lectura_ratio_mediana(self) -> float:
        ratios = [p.orden_lectura_ratio for p in self.paginas if p.orden_lectura_ratio is not None]
        if not ratios:
            return 1.0
        ratios.sort()
        mid = len(ratios) // 2
        if len(ratios) % 2:
            return ratios[mid]
        return (ratios[mid - 1] + ratios[mid]) / 2

    @property
    def usable(self) -> bool:
        """Regla documentada: usable si <5% de páginas sin texto y el orden
        de lectura mediano no se desvía más de 10% del orden espacial."""
        frac_sin_texto = len(self.paginas_sin_texto) / self.n_paginas if self.n_paginas else 1.0
        return frac_sin_texto < 0.05 and self.orden_lectura_ratio_mediana >= 0.90

    def resumen(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "paginas": self.n_paginas,
            "caracteres_totales": self.total_caracteres,
            "caracteres_prom_pagina": round(self.promedio_caracteres_pagina, 1),
            "paginas_sin_texto": len(self.paginas_sin_texto),
            "orden_lectura_mediana": round(self.orden_lectura_ratio_mediana, 3),
            "usable": self.usable,
        }


def _ratio_orden_lectura(page: "pdfplumber.page.Page") -> float | None:
    """Compara el orden de extracción de palabras (pdfplumber, orden del
    stream de contenido) contra el orden espacial (arriba->abajo,
    izquierda->derecha). Un ratio cercano a 1.0 significa que ambos órdenes
    coinciden, es decir, que el texto se puede leer linealmente tal como
    se extrae."""
    palabras = page.extract_words()
    if len(palabras) < 5:
        return None

    orden_extraccion = [w["text"] for w in palabras]
    orden_espacial = [
        w["text"]
        for w in sorted(palabras, key=lambda w: (round(w["top"] / 3), w["x0"]))
    ]
    return SequenceMatcher(a=orden_extraccion, b=orden_espacial).ratio()


def verificar_pdf(ruta: str | Path, doc_id: str) -> ReporteCalidad:
    ruta = Path(ruta)
    reader = PdfReader(str(ruta))
    n_paginas = len(reader.pages)

    reporte = ReporteCalidad(doc_id=doc_id, ruta=str(ruta), n_paginas=n_paginas)

    with pdfplumber.open(str(ruta)) as pdf:
        for i, (pypdf_page, plumber_page) in enumerate(zip(reader.pages, pdf.pages), start=1):
            texto = pypdf_page.extract_text() or ""
            tiene_texto = len(texto.strip()) >= MIN_CHARS_PARA_CONSIDERAR_CON_TEXTO
            ratio = _ratio_orden_lectura(plumber_page)
            reporte.paginas.append(
                ReportePagina(
                    pagina=i,
                    caracteres=len(texto),
                    tiene_texto=tiene_texto,
                    orden_lectura_ratio=ratio,
                )
            )

    return reporte

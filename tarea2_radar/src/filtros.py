r"""Fase 3 — separar condiciones NUMÉRICAS/TERRITORIALES (filtros) de la
parte SEMÁNTICA de una pregunta, antes de tocar el motor de embeddings.

## Por qué esto tiene que ser un filtro, no una similitud de embeddings

Una pregunta como "Obras de agua y alcantarillado en Cusco por encima de un
millón de soles" mezcla dos tipos de condición muy distintos:

1. **Condición semántica** ("obras de agua y alcantarillado"): no tiene un
   valor exacto que verificar — se trata de encontrar texto RELACIONADO en
   significado. Para esto los embeddings son exactamente la herramienta
   correcta.
2. **Condiciones territorial y numérica** ("en Cusco", "por encima de un
   millón"): SÍ tienen un valor exacto y verificable — el departamento es
   `CUSCO` o no lo es; el monto es mayor a 1,000,000 o no lo es. Un
   embedding no sabe hacer aritmética ni comparación exacta: la similitud
   semántica entre "un millón de soles" y un proceso de "980,000 soles" es
   casi idéntica a la de un proceso de "1,200,000 soles" — el texto es
   parecido, el número no cumple la condición. Confiarle esto a la
   similitud produciría procesos con montos por DEBAJO del umbral en los
   resultados (falsos positivos) y podría dejar fuera procesos válidos que
   describen el monto con otras palabras (falsos negativos). Lo mismo pasa
   con el departamento: un embedding puede traer procesos de Apurímac o
   Puno (semánticamente "cerca" de Cusco, misma región andina) aunque el
   departamento exacto no coincida.

Por eso: el departamento y el monto se aplican como **filtro exacto sobre
metadata** (`coleccion.query(..., where=...)` de ChromaDB) ANTES de rankear
por similitud, y solo el resto del texto de la pregunta se usa para la
búsqueda semántica. Esto no es una elección de implementación menor — es la
diferencia entre un resultado correcto y uno que "suena bien" pero está mal.

## Alcance de este parser (basado en reglas, no NLU completo)

Reconoce: los 25 departamentos oficiales (con o sin tilde), alias de
categoría (`obras`->works, `servicios`->services, `bienes`->goods), y
condiciones de monto con los patrones "por encima de/mayor a/más de X" y
"por debajo de/menor a/menos de X", donde X es un número en dígitos (con ,
o . como separador de miles) o la palabra "un"/"una"/"medio"/"media",
opcionalmente seguido de "mil" o "millón/millones". No intenta resolver
expresiones ambiguas, compuestas, ni números escritos en palabras más allá
de "un(a)"/"medio(a)" — alcance documentado, no una limitación oculta.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validar import DEPARTAMENTOS_OFICIALES, _clave_normalizada  # noqa: E402

# El orden de la alternancia de multiplicador IMPORTA: "mil" es prefijo de
# "millon(es)" (m-i-l...), así que si "mil" fuera la primera alternativa,
# una regex como `mil|millon(es)?` machearía solo "mil" dentro de
# "millones" y dejaría "lones" sin consumir. Por eso va primero la forma
# larga.
_PATRON_NUM = r"(?P<num{n}>[\d.,]+|un[a]?|medi[oa])\s*(?P<mult{n}>mill[oó]n(?:es)?|mil)?"
_PATRON_MONTO = re.compile(
    rf"(?:por encima de|mayor(?:es)? a|m[aá]s de|superior(?:es)? a)\s+{_PATRON_NUM.format(n=1)}"
    rf"|(?:por debajo de|menor(?:es)? a|menos de|inferior(?:es)? a)\s+{_PATRON_NUM.format(n=2)}",
    re.IGNORECASE,
)

_MULTIPLICADORES = {"MIL": 1_000, "MILLON": 1_000_000, "MILLONES": 1_000_000}
_NUMEROS_EN_PALABRAS = {"UN": 1.0, "UNA": 1.0, "MEDIO": 0.5, "MEDIA": 0.5}


def _multiplicador(texto: str | None) -> int:
    if not texto:
        return 1
    return _MULTIPLICADORES.get(_clave_normalizada(texto), 1)


def _numero(texto: str) -> float:
    clave = _clave_normalizada(texto)
    if clave in _NUMEROS_EN_PALABRAS:
        return _NUMEROS_EN_PALABRAS[clave]
    return _numero_a_float(texto)


def _numero_a_float(texto_num: str) -> float:
    # "1,000,000" o "1.000.000" o "1000000" -> 1000000.0 ; "1.5" -> 1.5
    limpio = texto_num.replace(",", "")
    # si tiene un solo punto y termina en 3 digitos tras el, probablemente es separador de miles (es-PE a veces usa '.')
    if limpio.count(".") == 1 and len(limpio.split(".")[1]) == 3:
        limpio = limpio.replace(".", "")
    return float(limpio)


def _contiene_como_palabra(clave_texto: str, clave_buscada: str) -> bool:
    """Busca `clave_buscada` como PALABRA completa dentro de `clave_texto`,
    no como substring suelto. Bug real encontrado en evaluación: una
    búsqueda por substring hacía que 'ICA' (departamento) machease dentro
    de 'ELECTRONICAS' ('...electrón-ICA-s'), metiendo un filtro de
    departamento incorrecto en una pregunta que no lo pedía."""
    return re.search(rf"\b{re.escape(clave_buscada)}\b", clave_texto) is not None


def _extraer_departamento(texto: str) -> str | None:
    clave_texto = _clave_normalizada(texto)
    for depto in DEPARTAMENTOS_OFICIALES:
        if _contiene_como_palabra(clave_texto, _clave_normalizada(depto)):
            return depto
    return None


def _extraer_categoria(texto: str, alias: dict[str, str]) -> str | None:
    clave_texto = _clave_normalizada(texto)
    for alias_texto, categoria in alias.items():
        if _contiene_como_palabra(clave_texto, _clave_normalizada(alias_texto)):
            return categoria
    return None


def _extraer_monto(texto: str) -> dict | None:
    m = _PATRON_MONTO.search(texto)
    if not m:
        return None
    if m.group("num1") is not None:
        base = _numero(m.group("num1"))
        return {"monto": {"$gt": base * _multiplicador(m.group("mult1"))}}
    base = _numero(m.group("num2"))
    return {"monto": {"$lt": base * _multiplicador(m.group("mult2"))}}


def extraer_filtros(pregunta: str, alias_categoria: dict[str, str]) -> dict | None:
    """Devuelve un `where` de ChromaDB con las condiciones detectadas, o
    None si no se detectó ninguna condición estructurada (la pregunta es
    puramente semántica)."""
    condiciones = []

    depto = _extraer_departamento(pregunta)
    if depto:
        condiciones.append({"departamento": {"$eq": depto}})

    categoria = _extraer_categoria(pregunta, alias_categoria)
    if categoria:
        condiciones.append({"categoria": {"$eq": categoria}})

    monto = _extraer_monto(pregunta)
    if monto:
        condiciones.append(monto)

    if not condiciones:
        return None
    if len(condiciones) == 1:
        return condiciones[0]
    return {"$and": condiciones}

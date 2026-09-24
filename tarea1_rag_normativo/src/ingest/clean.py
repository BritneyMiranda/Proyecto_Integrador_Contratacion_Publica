"""Limpieza de texto extraído de PDFs de normativa peruana.

Regla documentada
------------------
Los PDF de El Peruano repiten, en (casi) todas las páginas, el mismo
encabezado/pie (número de página, nombre de sección, fecha de edición)
pegado directamente al cuerpo del texto porque `pypdf` extrae en orden de
stream, no por bloques visuales.

En vez de adivinar el texto exacto del encabezado (frágil: cambia de
documento a documento), la regla es estructural:

    Cualquier línea cuya versión "normalizada" (dígitos quitados, espacios
    quitados) aparece en al menos `min_fraction` de las páginas del
    documento se considera ruido de encabezado/pie y se elimina.

Los espacios también se normalizan porque El Peruano maqueta en formato
periódico: el encabezado sale con espaciado distinto en páginas pares e
impares (comprobado en `ds_001_2026_ef.pdf`: la página 41 extrae
`"41NORMAS LEGALESJueves 8 de..."` sin espacio tras el número, mientras que
la página 42 extrae `"42 NORMAS LEGALES Jueves 8 de..."` con espacio). Sin
ignorar los espacios, cada variante se cuenta por separado y ninguna llega
al umbral de frecuencia.

Esto es seguro porque el contenido legal (artículos, incisos, listas con
letras) no se repite letra por letra entre páginas -> nunca cae en el
umbral de frecuencia y por lo tanto nunca se borra por accidente. Los
marcadores de artículo ("Artículo 5.-") y de lista ("a)", "b)") se
conservan siempre.

El umbral (`min_fraction`) y el patrón de número de página suelto se
calibran con evidencia real: se corre primero `detectar_lineas_repetidas`
y se inspecciona la lista antes de aplicar `limpiar_pagina` a todo el
documento.
"""

from __future__ import annotations

import re
from collections import Counter

_RE_DIGITOS = re.compile(r"\d+")
_RE_SOLO_NUMERO = re.compile(r"^\s*\d{1,4}\s*$")
_RE_ESPACIOS = re.compile(r"[ \t]+")
_RE_TODO_ESPACIO = re.compile(r"\s+")
_RE_LINEAS_VACIAS = re.compile(r"\n{3,}")


def _normalizar_linea(linea: str) -> str:
    """Clave de comparación para detectar boilerplate: sin dígitos y sin
    espacios, para que no importe ni el número de página ni el espaciado
    par/impar de la maquetación periodística."""
    sin_digitos = _RE_DIGITOS.sub("", linea.strip())
    return _RE_TODO_ESPACIO.sub("", sin_digitos).lower()


def detectar_lineas_repetidas(
    paginas_texto: list[str],
    min_fraction: float = 0.7,
    max_len_linea: int = 90,
) -> set[str]:
    """Devuelve el conjunto de líneas (normalizadas) que se repiten en al
    menos `min_fraction` de las páginas -> candidatas a encabezado/pie."""
    n_paginas = len(paginas_texto)
    if n_paginas == 0:
        return set()

    apariciones: Counter[str] = Counter()
    for texto in paginas_texto:
        lineas_normalizadas_en_pagina = {
            _normalizar_linea(linea)
            for linea in texto.splitlines()
            if linea.strip()
        }
        apariciones.update(lineas_normalizadas_en_pagina)

    umbral = max(2, int(n_paginas * min_fraction))
    return {
        linea
        for linea, veces in apariciones.items()
        if veces >= umbral and len(linea) < max_len_linea
    }


def limpiar_pagina(texto: str, lineas_boilerplate: set[str]) -> str:
    """Elimina líneas de boilerplate y números de página sueltos; conserva
    todo lo demás intacto (incluidos artículos y listas con letras)."""
    lineas_limpias = []
    for linea in texto.splitlines():
        cruda = linea.strip()
        if not cruda:
            lineas_limpias.append("")
            continue
        if _RE_SOLO_NUMERO.match(cruda):
            continue
        if _normalizar_linea(cruda) in lineas_boilerplate:
            continue
        lineas_limpias.append(_RE_ESPACIOS.sub(" ", cruda))

    texto_unido = "\n".join(lineas_limpias)
    texto_unido = _RE_LINEAS_VACIAS.sub("\n\n", texto_unido)
    return texto_unido.strip()

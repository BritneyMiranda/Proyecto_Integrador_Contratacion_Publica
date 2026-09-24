"""Precios de los proveedores de generación soportados — verificados, no de
memoria. El motor elige el proveedor por `config.yaml` (`generacion.proveedor`).

## Gemini (proveedor activo por defecto — capa gratuita)

Fuente: https://ai.google.dev/gemini-api/docs/pricing (página oficial),
verificado el 2026-09-20. La propia página lo dice explícitamente: el precio
es plano ("Pricing is flat"), sin ninguna variación por hora del día — a
diferencia de DeepSeek (ver abajo). Documentamos esto explícitamente porque
el enunciado pide investigar si el proveedor varía precio por hora: para
Gemini, la respuesta verificada es que NO varía.

Usamos `gemini-2.5-flash` en la capa gratuita de Google AI Studio
(`generacion.nivel: gratuito` en config.yaml): mientras la cuenta no tenga
facturación habilitada, el costo real es $0.00 por llamada, sujeto a los
límites de cuota de peticiones/minuto de esa capa (no hay un número fijo
publicado que podamos citar con confianza — cambia con el tiempo — así que
no lo inventamos aquí). Si `nivel: pagado`, se cobra con la tabla de abajo.

## DeepSeek (proveedor alterno, ya probado con la API real)

Fuente: https://api-docs.deepseek.com/quick_start/pricing/ (página oficial),
corroborado por búsqueda independiente. Fecha de verificación: 2026-09-20.
DeepSeek reemplazó su antiguo descuento nocturno fijo por un esquema
peak/off-peak el 16 de agosto de 2026: las horas "peak" cuestan EL DOBLE que
las "off-peak" (off-peak = mitad del precio peak).
Horas "peak": 01:00-04:00 y 06:00-10:00 UTC, lunes a viernes. Todo lo demás
(fines de semana incluidos) es "off-peak". No filtramos feriados chinos
específicos — limitación conocida y documentada.
"""

from __future__ import annotations

from datetime import datetime, timezone

FUENTE_PRECIOS_GEMINI = "https://ai.google.dev/gemini-api/docs/pricing"
FECHA_VERIFICACION_GEMINI = "2026-09-20"

# USD por 1,000,000 de tokens (tarifa estándar/pagada; la capa gratuita cobra $0).
PRECIOS_GEMINI_USD_POR_1M_TOKENS = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}

FUENTE_PRECIOS_DEEPSEEK = "https://api-docs.deepseek.com/quick_start/pricing/"
FECHA_VERIFICACION_DEEPSEEK = "2026-09-20"

PRECIOS_DEEPSEEK_USD_POR_1M_TOKENS = {
    "deepseek-flash": {
        "input_cache_hit": {"off_peak": 0.003, "peak": 0.006},
        "input_cache_miss": {"off_peak": 0.15, "peak": 0.30},
        "output": {"off_peak": 0.60, "peak": 1.20},
    },
    "deepseek-v4-pro": {
        "input_cache_hit": {"off_peak": 0.022, "peak": 0.044},
        "input_cache_miss": {"off_peak": 0.66, "peak": 1.32},
        "output": {"off_peak": 1.98, "peak": 3.96},
    },
}

_VENTANAS_PEAK_UTC = [(1 * 60, 4 * 60), (6 * 60, 10 * 60)]  # (inicio, fin) en minutos desde medianoche
_DIAS_PEAK = {0, 1, 2, 3, 4}  # lunes=0 ... viernes=4


def es_hora_peak(momento_utc: datetime) -> bool:
    """Solo aplica a DeepSeek. Gemini tiene precio plano — ver docstring."""
    if momento_utc.tzinfo is None:
        momento_utc = momento_utc.replace(tzinfo=timezone.utc)
    momento_utc = momento_utc.astimezone(timezone.utc)

    if momento_utc.weekday() not in _DIAS_PEAK:
        return False
    minutos_del_dia = momento_utc.hour * 60 + momento_utc.minute
    return any(inicio <= minutos_del_dia < fin for inicio, fin in _VENTANAS_PEAK_UTC)


def calcular_costo_gemini(modelo: str, tokens_entrada: int, tokens_salida: int, nivel: str) -> dict:
    if nivel == "gratuito":
        costo = 0.0
    else:
        tabla = PRECIOS_GEMINI_USD_POR_1M_TOKENS[modelo]
        costo = tokens_entrada / 1_000_000 * tabla["input"] + tokens_salida / 1_000_000 * tabla["output"]

    return {
        "costo_usd": round(costo, 8),
        "franja_horaria": "no_aplica",  # Gemini no varía precio por hora — verificado, no un vacío por omisión
        "fuente_precios": FUENTE_PRECIOS_GEMINI,
        "fecha_verificacion_precios": FECHA_VERIFICACION_GEMINI,
    }


def calcular_costo_deepseek(
    modelo: str,
    tokens_entrada_cache_hit: int,
    tokens_entrada_cache_miss: int,
    tokens_salida: int,
    momento_utc: datetime,
) -> dict:
    franja = "peak" if es_hora_peak(momento_utc) else "off_peak"
    tabla = PRECIOS_DEEPSEEK_USD_POR_1M_TOKENS[modelo]

    costo = (
        tokens_entrada_cache_hit / 1_000_000 * tabla["input_cache_hit"][franja]
        + tokens_entrada_cache_miss / 1_000_000 * tabla["input_cache_miss"][franja]
        + tokens_salida / 1_000_000 * tabla["output"][franja]
    )

    return {
        "costo_usd": round(costo, 8),
        "franja_horaria": franja,
        "fuente_precios": FUENTE_PRECIOS_DEEPSEEK,
        "fecha_verificacion_precios": FECHA_VERIFICACION_DEEPSEEK,
    }

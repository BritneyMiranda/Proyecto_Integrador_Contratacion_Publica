"""Registro observable de cada llamada al modelo de generación.

Cada llamada (exitosa o fallida) agrega una fila a `logs/costos.csv`. El
motor nunca decide si loggear o no: siempre registra, incluso cuando la
API falla (con costo 0 y exito=False), porque el requisito es poder
auditar todas las llamadas, no solo las que salieron bien.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_LOG = RAIZ_PROYECTO / "logs" / "costos.csv"

_CAMPOS = [
    "timestamp_utc",
    "modelo",
    "franja_horaria",
    "tokens_entrada_cache_hit",
    "tokens_entrada_cache_miss",
    "tokens_salida",
    "latencia_ms",
    "costo_usd",
    "exito",
    "detalle_error",
    "fuente_precios",
    "fecha_verificacion_precios",
]


@dataclass
class RegistroLlamada:
    timestamp_utc: str
    modelo: str
    franja_horaria: str
    tokens_entrada_cache_hit: int
    tokens_entrada_cache_miss: int
    tokens_salida: int
    latencia_ms: float
    costo_usd: float
    exito: bool
    detalle_error: str = ""
    fuente_precios: str = ""
    fecha_verificacion_precios: str = ""


def registrar_llamada(registro: RegistroLlamada, ruta_log: Path = RUTA_LOG) -> None:
    ruta_log.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta_log.exists()
    with ruta_log.open("a", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=_CAMPOS)
        if not existe:
            escritor.writeheader()
        escritor.writerow(asdict(registro))


def ahora_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

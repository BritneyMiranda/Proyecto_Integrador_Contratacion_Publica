r"""Fase 1 de la Tarea 2 — adquisición de datos abiertos de contratación (OECE).

La API pública documentada en la SPA de OECE (descubierta inspeccionando el
bundle de Angular servido en contratacionesabiertas.oece.gob.pe — la página
"/descargas" es una SPA que no expone enlaces de descarga estáticos, así que
se ubicó el endpoint real que ella misma consume) expone:

- `GET /api/v1/files` (paginado): lista de archivos mensuales masivos
  disponibles, por fuente (`seace_v3`, y fuentes históricas `seace_v2`/`seace_v1`)
  y por mes, en CSV/XLSX/JSON.
- `GET /api/v1/file/<fuente>/<formato>/<año>/<mes>/`: descarga el archivo
  mensual como ZIP.
- `GET /api/v1/releases` y `GET /api/v1/records`: API "en vivo" de
  publicaciones OCDS — para obtener actualizaciones recientes, no para
  descargar el histórico completo (eso es lo que son los archivos masivos).

Diseño de esta descarga (todos los requisitos de la Fase 1):
- **Nunca pierde trabajo ya hecho**: cada archivo se descarga primero a una
  ruta temporal y solo se renombra al destino final si la descarga terminó
  completa; si algo falla a mitad de camino, el archivo bueno anterior (si
  existía) queda intacto.
- **Nunca re-descarga lo que ya existe**: si el ZIP de destino ya existe en
  disco, se salta (idempotente — se puede volver a correr sin costo).
- **Limitación de solicitudes**: pausa entre peticiones a la API.
- **Manejo de errores con reintento**: backoff exponencial ante fallos de
  red o respuestas 5xx.
- **Se registra**: tiempo transcurrido, número de solicitudes y tamaño de
  cada archivo, en `logs/adquisicion.csv`.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_API = "https://contratacionesabiertas.oece.gob.pe/api/v1"
SEGUNDOS_ENTRE_PETICIONES = 1.0
REINTENTOS_MAXIMOS = 4

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_LOG = RAIZ_PROYECTO / "logs" / "adquisicion.csv"

_CAMPOS_LOG = [
    "timestamp_utc",
    "operacion",
    "url",
    "intentos",
    "duracion_s",
    "tamano_bytes",
    "exito",
    "detalle_error",
]

_contador_peticiones = 0


def _registrar(operacion: str, url: str, intentos: int, duracion_s: float, tamano_bytes: int, exito: bool, error: str = "") -> None:
    RUTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    existe = RUTA_LOG.exists()
    with RUTA_LOG.open("a", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=_CAMPOS_LOG)
        if not existe:
            escritor.writeheader()
        escritor.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "operacion": operacion,
                "url": url,
                "intentos": intentos,
                "duracion_s": round(duracion_s, 2),
                "tamano_bytes": tamano_bytes,
                "exito": exito,
                "detalle_error": error,
            }
        )


def _peticion_con_reintento(url: str, operacion: str, **kwargs) -> requests.Response:
    """GET con backoff exponencial. Cuenta la petición para el límite de
    solicitudes y registra el resultado (éxito o fracaso) siempre."""
    global _contador_peticiones

    ultimo_error = None
    for intento in range(1, REINTENTOS_MAXIMOS + 1):
        _contador_peticiones += 1
        inicio = time.monotonic()
        try:
            respuesta = requests.get(url, timeout=60, **kwargs)
            duracion = time.monotonic() - inicio
            if respuesta.status_code == 200:
                _registrar(operacion, url, intento, duracion, len(respuesta.content), True)
                time.sleep(SEGUNDOS_ENTRE_PETICIONES)
                return respuesta
            ultimo_error = f"HTTP {respuesta.status_code}"
        except requests.RequestException as err:
            duracion = time.monotonic() - inicio
            ultimo_error = str(err)

        _registrar(operacion, url, intento, duracion, 0, False, ultimo_error)
        if intento < REINTENTOS_MAXIMOS:
            time.sleep(2**intento)  # backoff: 2s, 4s, 8s...

    raise RuntimeError(f"Fallaron los {REINTENTOS_MAXIMOS} intentos para {url}: {ultimo_error}")


def listar_archivos_mensuales(fuente: str = "seace_v3", max_paginas: int = 10) -> list[dict]:
    """Recorre `GET /api/v1/files` (paginado) y devuelve los metadatos de
    los archivos mensuales de la fuente pedida."""
    resultados = []
    pagina = 1
    while pagina <= max_paginas:
        url = f"{BASE_API}/files?page={pagina}"
        respuesta = _peticion_con_reintento(url, operacion="listar_archivos")
        cuerpo = respuesta.json()
        resultados.extend(r for r in cuerpo["results"] if r["source"] == fuente)
        if not cuerpo["pagination"]["has_next"]:
            break
        pagina += 1
    return resultados


def descargar_mes(
    fuente: str,
    anio: str,
    mes: str,
    dir_destino: Path = RAIZ_PROYECTO / "data" / "raw",
    formato: str = "json",
) -> Path:
    """Descarga el ZIP mensual. Si ya existe en disco, no hace ninguna
    petición (idempotente). Escribe primero a un archivo temporal y solo
    renombra al destino final si la descarga se completó — una descarga
    fallida nunca deja un archivo corrupto pisando uno bueno anterior."""
    dir_destino.mkdir(parents=True, exist_ok=True)
    ruta_final = dir_destino / f"{anio}-{mes}_{fuente}_{formato}.zip"

    if ruta_final.exists():
        print(f"  ya existe, no se vuelve a descargar: {ruta_final.name}")
        return ruta_final

    url = f"{BASE_API}/file/{fuente}/{formato}/{anio}/{mes}/"
    inicio = time.monotonic()
    respuesta = _peticion_con_reintento(url, operacion="descargar_mes")
    duracion = time.monotonic() - inicio

    ruta_temporal = ruta_final.with_suffix(".zip.tmp")
    ruta_temporal.write_bytes(respuesta.content)
    ruta_temporal.rename(ruta_final)  # atómico: solo aparece el archivo final si todo salió bien

    print(f"  descargado: {ruta_final.name} ({len(respuesta.content):,} bytes, {duracion:.1f}s)")
    return ruta_final


def obtener_peticiones_realizadas() -> int:
    return _contador_peticiones


if __name__ == "__main__":
    print("=== Fase 1 Tarea 2: adquisición ===")
    inicio_total = time.monotonic()

    MESES_OBJETIVO = [("2026", "06"), ("2026", "07"), ("2026", "08")]

    for anio, mes in MESES_OBJETIVO:
        print(f"\nMes {anio}-{mes}:")
        descargar_mes("seace_v3", anio, mes)

    print(f"\nPeticiones HTTP totales: {obtener_peticiones_realizadas()}")
    print(f"Tiempo total: {time.monotonic() - inicio_total:.1f} s")
    print(f"Log de adquisición: {RUTA_LOG}")

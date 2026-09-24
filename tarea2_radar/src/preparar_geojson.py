r"""Fase 4 — prepara el GeoJSON de departamentos para el panel (un solo run,
no se repite al cargar la página).

Fuente: geoBoundaries (ADM1, Perú) — CC-BY 4.0, citable, sin bloqueo de
descarga. `data/raw/peru_departamentos.geojson`, 26 features:
geoBoundaries separa "Municipalidad Metropolitana de Lima" de "Lima" como
dos unidades ADM1 distintas, pero en nuestro corpus (campo `departamento`
de OCDS) todo lo de Lima Metropolitana cae bajo `departamento=LIMA` — así
que se fusionan geometrías para que el panel tenga exactamente los 25
departamentos oficiales, ni más ni menos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ_TAREA2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_TAREA2 / "src"))

from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from validar import normalizar_departamento

RUTA_ENTRADA = RAIZ_TAREA2 / "data" / "raw" / "peru_departamentos.geojson"
RUTA_SALIDA = RAIZ_TAREA2 / "data" / "processed" / "departamentos.geojson"

# geoBoundaries usa "El Callao" (con artículo) y separa "Municipalidad
# Metropolitana de Lima" de "Lima" como dos unidades ADM1 distintas; en
# nuestro corpus (campo `departamento` de OCDS) ambas caen bajo LIMA.
_ALIAS_MANUAL = {"El Callao": "Callao", "Municipalidad Metropolitana de Lima": "Lima"}


def preparar() -> None:
    with open(RUTA_ENTRADA, encoding="utf-8") as fh:
        gj = json.load(fh)

    geometrias_por_departamento: dict[str, list] = {}
    for feature in gj["features"]:
        nombre_crudo = feature["properties"]["shapeName"]
        nombre_crudo = _ALIAS_MANUAL.get(nombre_crudo, nombre_crudo)
        oficial = normalizar_departamento(nombre_crudo)
        if oficial is None:
            print(f"AVISO: '{nombre_crudo}' no se pudo normalizar a un departamento oficial — se omite")
            continue
        geometrias_por_departamento.setdefault(oficial, []).append(shape(feature["geometry"]))

    nuevas_features = []
    for departamento, geoms in geometrias_por_departamento.items():
        geometria_unida = unary_union(geoms) if len(geoms) > 1 else geoms[0]
        nuevas_features.append(
            {
                "type": "Feature",
                "properties": {"departamento": departamento},
                "geometry": mapping(geometria_unida),
            }
        )

    geojson_final = {"type": "FeatureCollection", "features": nuevas_features}

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as fh:
        json.dump(geojson_final, fh, ensure_ascii=False)

    print(f"{len(nuevas_features)} departamentos escritos en {RUTA_SALIDA}")
    from validar import DEPARTAMENTOS_OFICIALES
    print("Departamentos oficiales sin geometría:", set(DEPARTAMENTOS_OFICIALES) - set(geometrias_por_departamento))


if __name__ == "__main__":
    preparar()

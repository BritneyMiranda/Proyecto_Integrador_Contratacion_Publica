"""Interfaz de terminal para el motor RAG.

Ejemplo de "múltiples interfaces, un solo motor": este archivo NO tiene
ninguna lógica de RAG propia, solo llama a `engine.motor.responder()` y
muestra el resultado. La interfaz Streamlit (cuando se agregue) hará
exactamente lo mismo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
RAIZ_PROYECTO = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAIZ_PROYECTO / ".env")

from engine.motor import ErrorDeAPI, cargar_config, responder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asistente RAG de contratación pública (Ley N.° 32069 + DS N.° 001-2026-EF)."
    )
    parser.add_argument(
        "pregunta",
        nargs="*",
        help="La pregunta a responder. Si se omite, se pide de forma interactiva.",
    )
    args = parser.parse_args()

    pregunta = " ".join(args.pregunta).strip()
    if not pregunta:
        pregunta = input("Pregunta: ").strip()
    if not pregunta:
        print("No escribiste ninguna pregunta.", file=sys.stderr)
        sys.exit(1)

    config = cargar_config()

    try:
        resultado = responder(pregunta, config)
    except ErrorDeAPI as err:
        print(f"\n[ERROR DE API] proveedor={err.proveedor} status_code={err.status_code}", file=sys.stderr)
        print(f"  {err}", file=sys.stderr)
        sys.exit(2)

    print(f"\nPregunta: {pregunta}\n")

    if resultado["abstuvo"]:
        print("[ABSTENCION] La pregunta cae fuera del umbral de similitud del corpus indexado.\n")

    print("Respuesta:")
    print(resultado["respuesta"])

    print("\nFuentes:")
    for f in resultado["fuentes"]:
        print(f"  - {f['titulo']} (versión {f['version']}), página {f['pagina']} — similitud {f['similitud']}")

    print(f"\nmejor_similitud={resultado['mejor_similitud']}  umbral_similitud={resultado['umbral_similitud']}")
    if not resultado["abstuvo"]:
        print(f"tokens_entrada={resultado['tokens_entrada']}  tokens_salida={resultado['tokens_salida']}")
        print(
            f"costo_usd={resultado['costo_usd']}  franja_horaria={resultado['franja_horaria']}  "
            f"latencia_ms={resultado['latencia_ms']}"
        )


if __name__ == "__main__":
    main()

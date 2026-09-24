"""Embeddings locales con `intfloat/multilingual-e5-small` (CPU-only).

Hechos verificados contra el propio modelo/model card (no supuestos):

- `max_seq_length` / `tokenizer.model_max_length` = 512 tokens — confirmado
  cargando el modelo (`SentenceTransformer(...).max_seq_length`).
- El README oficial del repo en HuggingFace dice, literal:
  "Each input text should start with 'query: ' or 'passage: ', even for
  non-English texts." -> por eso `E5Embedder` antepone `"query: "` a las
  preguntas y `"passage: "` a los fragmentos que se indexan.
- Dimensión del embedding: 384.

Relación con el tamaño de fragmento (ver `chunking.py` y el notebook de
Fase 2): las configuraciones evaluadas usan como máximo 256 tokens por
fragmento, muy por debajo del límite de 512 — así el prefijo y los tokens
especiales nunca empujan al modelo a truncar contenido.

## Interfaz común (Fase 4)

`crear_embedder(proveedor, modo)` devuelve una `EmbeddingFunction` de
ChromaDB — local (`E5Embedder`) o de API (`OpenAIEmbedder`, modelo
`text-embedding-3-small`). Ambas exponen la misma interfaz
(`__call__(input) -> Embeddings`), así que cambiar de proveedor es cambiar
`embeddings.proveedor` en `config.yaml`, no tocar código que las use
(`vectorstore.py`, `motor.py`, `eval/evaluar.py`).

Precio de `text-embedding-3-small` verificado el 2026-09-20 contra la
documentación oficial de OpenAI (`developers.openai.com/api/docs/models/
text-embedding-3-small`): $0.02 USD por 1M tokens. La dimensión del vector
no está documentada en esa página — se toma directamente de la respuesta
real de la API (`len(response.data[0].embedding)`), no se supone.
"""

from __future__ import annotations

import os

from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

NOMBRE_MODELO = "intfloat/multilingual-e5-small"

NOMBRE_MODELO_OPENAI = "text-embedding-3-small"
PRECIO_OPENAI_USD_POR_1M_TOKENS = 0.02
FUENTE_PRECIO_OPENAI = "https://developers.openai.com/api/docs/models/text-embedding-3-small"
FECHA_VERIFICACION_PRECIO_OPENAI = "2026-09-20"

# Usado en la comparación real de Fase 4 en vez de OpenAI: la cuenta de
# OpenAI disponible no tenía saldo prepago (OpenAI no tiene capa gratuita
# para su API, ni de prueba) y Gemini sí, así que se sustituyó el
# proveedor "API" por Gemini para poder correr números reales, no
# estimados. Precio verificado el 2026-09-20 contra
# ai.google.dev/gemini-api/docs/pricing: capa gratuita = $0 real; tarifa
# pagada de referencia = $0.20 por 1M tokens de texto.
NOMBRE_MODELO_GEMINI = "gemini-embedding-2"
PRECIO_GEMINI_USD_POR_1M_TOKENS_PAGADO = 0.20
FUENTE_PRECIO_GEMINI = "https://ai.google.dev/gemini-api/docs/pricing"
FECHA_VERIFICACION_PRECIO_GEMINI = "2026-09-20"

_modelo: SentenceTransformer | None = None


def cargar_modelo() -> SentenceTransformer:
    global _modelo
    if _modelo is None:
        _modelo = SentenceTransformer(NOMBRE_MODELO)
    return _modelo


def info_modelo() -> dict:
    modelo = cargar_modelo()
    try:
        dimension = modelo.get_embedding_dimension()
    except AttributeError:
        dimension = modelo.get_sentence_embedding_dimension()
    return {
        "nombre": NOMBRE_MODELO,
        "max_seq_length_tokens": modelo.max_seq_length,
        "dimension_embedding": dimension,
        "prefijo_query": "query: ",
        "prefijo_passage": "passage: ",
        "fuente": "model card oficial en HuggingFace (README.md de intfloat/multilingual-e5-small)",
    }


class E5Embedder(EmbeddingFunction):
    """EmbeddingFunction compatible con ChromaDB. `modo` decide el
    prefijo obligatorio del modelo: 'passage' al indexar fragmentos,
    'query' al vectorizar la pregunta del usuario."""

    def __init__(self, modo: str = "passage"):
        if modo not in ("passage", "query"):
            raise ValueError("modo debe ser 'passage' o 'query'")
        self.modo = modo

    def __call__(self, input: Documents) -> Embeddings:
        modelo = cargar_modelo()
        prefijo = f"{self.modo}: "
        textos = [prefijo + t for t in input]
        vectores = modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False)
        return vectores.tolist()

    def name(self) -> str:
        return f"{NOMBRE_MODELO}::{self.modo}"


class OpenAIEmbedder(EmbeddingFunction):
    """EmbeddingFunction compatible con ChromaDB, vía la API de OpenAI.
    `text-embedding-3-small` no distingue query/passage con un prefijo
    obligatorio como E5 — es de propósito general — así que `modo` se
    acepta por simetría de interfaz pero no cambia el texto enviado.

    Registra `ultimo_uso_tokens` (última llamada) y acumula
    `tokens_acumulados` (todas las llamadas hechas con esta instancia) para
    poder medir el costo real — no una estimación — de indexar/consultar
    todo un lote, en la comparación de Fase 4."""

    def __init__(self, modo: str = "passage", modelo: str = NOMBRE_MODELO_OPENAI):
        self.modo = modo
        self.modelo = modelo
        self.ultimo_uso_tokens = 0
        self.tokens_acumulados = 0
        self._cliente = None

    def _obtener_cliente(self):
        if self._cliente is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("Falta OPENAI_API_KEY en el entorno (.env)")
            self._cliente = OpenAI(api_key=api_key)
        return self._cliente

    def __call__(self, input: Documents) -> Embeddings:
        respuesta = self._obtener_cliente().embeddings.create(model=self.modelo, input=list(input))
        self.ultimo_uso_tokens = respuesta.usage.total_tokens
        self.tokens_acumulados += respuesta.usage.total_tokens
        return [d.embedding for d in respuesta.data]

    def name(self) -> str:
        return f"{self.modelo}::{self.modo}"


class GeminiEmbedder(EmbeddingFunction):
    """EmbeddingFunction compatible con ChromaDB, vía la API de Gemini.
    A diferencia de OpenAI, Gemini sí distingue `task_type` para
    consultas ('RETRIEVAL_QUERY') vs. pasajes ('RETRIEVAL_DOCUMENT') —
    mismo patrón que E5, pero con nombres de la propia API de Gemini en
    vez de un prefijo de texto.

    La capa gratuita limita a 100 peticiones de `embed_content` por
    minuto (visto en vivo: error 429 RESOURCE_EXHAUSTED indexando el
    corpus completo de un jalón). Por eso cada llamada se espacia con un
    `time.sleep` y, si aun así se topa con el límite, reintenta leyendo
    el tiempo de espera que la propia API sugiere en el mensaje de error."""

    _SEGUNDOS_ENTRE_LLAMADAS = 0.65  # ~92/min, bajo el límite de 100/min
    _REINTENTOS_MAXIMOS = 6

    def __init__(self, modo: str = "passage", modelo: str = NOMBRE_MODELO_GEMINI, espaciar_llamadas: bool = True):
        if modo not in ("passage", "query"):
            raise ValueError("modo debe ser 'passage' o 'query'")
        self.modo = modo
        self.modelo = modelo
        # El freno solo debe aplicarse al INDEXAR en volumen (cientos de
        # fragmentos seguidos, ahí sí se topa con el límite de 100/min).
        # Al medir latencia de UNA consulta para el reporte de Fase 4, no
        # se frena: si no, la métrica reportada sería nuestro propio
        # `time.sleep`, no la latencia real de la API.
        self.espaciar_llamadas = espaciar_llamadas
        self._cliente = None

    def _obtener_cliente(self):
        if self._cliente is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise RuntimeError("Falta GEMINI_API_KEY en el entorno (.env)")
            self._cliente = genai.Client(api_key=api_key)
        return self._cliente

    def _embeber_uno(self, texto: str, task_type: str, cliente) -> list[float]:
        import re
        import time

        from google.genai import errors as genai_errors
        from google.genai import types

        for intento in range(self._REINTENTOS_MAXIMOS):
            try:
                respuesta = cliente.models.embed_content(
                    model=self.modelo,
                    contents=texto,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                if self.espaciar_llamadas:
                    time.sleep(self._SEGUNDOS_ENTRE_LLAMADAS)
                return respuesta.embeddings[0].values
            except genai_errors.ClientError as err:
                if "RESOURCE_EXHAUSTED" not in str(err) or intento == self._REINTENTOS_MAXIMOS - 1:
                    raise
                coincidencia = re.search(r"retry in ([\d.]+)s", str(err))
                espera = float(coincidencia.group(1)) + 3 if coincidencia else 60.0
                time.sleep(espera)
        raise RuntimeError("No debería llegar aquí")  # pragma: no cover

    def __call__(self, input: Documents) -> Embeddings:
        task_type = "RETRIEVAL_DOCUMENT" if self.modo == "passage" else "RETRIEVAL_QUERY"
        cliente = self._obtener_cliente()
        return [self._embeber_uno(texto, task_type, cliente) for texto in input]

    def name(self) -> str:
        return f"{self.modelo}::{self.modo}"


def crear_embedder(proveedor: str, modo: str = "passage") -> EmbeddingFunction:
    """Interfaz común de fábrica: cambiar de modelo de embeddings es
    cambiar `embeddings.proveedor` en config.yaml, no tocar código."""
    if proveedor == "local":
        return E5Embedder(modo=modo)
    if proveedor == "openai":
        return OpenAIEmbedder(modo=modo)
    if proveedor == "gemini":
        return GeminiEmbedder(modo=modo)
    raise ValueError(f"Proveedor de embeddings desconocido: {proveedor!r}")

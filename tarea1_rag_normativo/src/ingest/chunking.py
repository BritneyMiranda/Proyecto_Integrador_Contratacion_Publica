"""Segmentación (chunking) de páginas limpias en fragmentos.

El conteo de longitud usa el tokenizer REAL del modelo de embeddings
(`intfloat/multilingual-e5-small`, ver `embeddings.py`), no un proxy como
`tiktoken`, porque lo que importa es no acercarse al límite de 512 tokens
de ESE modelo.

Cada fragmento cae dentro de una sola página: nunca se junta texto de dos
páginas, así el campo `pagina` de cada fragmento es siempre exacto para
citar. El `chunk_id` es determinístico:

    "{doc_id}::p{pagina:04d}::c{índice_dentro_de_la_página:03d}"

-> único entre documentos (prefijo `doc_id`) y estable entre ejecuciones,
siempre que no cambien el texto de entrada ni los parámetros de chunking
(mismo texto + mismo `chunk_size`/`chunk_overlap` -> mismos fragmentos en
el mismo orden, porque `RecursiveCharacterTextSplitter` es determinístico).
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import cargar_modelo

_SEPARADORES = ["\n\n", "\n", ". ", "; ", " ", ""]


def contar_tokens(texto: str) -> int:
    tokenizer = cargar_modelo().tokenizer
    return len(tokenizer.encode(texto, add_special_tokens=False))


def _construir_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=contar_tokens,
        separators=_SEPARADORES,
    )


def segmentar_paginas(
    registros_pagina: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """`registros_pagina`: registros tal como salen de data/processed/<doc>.jsonl
    (doc_id, titulo, institucion, tipo, fuente_url, version, pagina, texto).
    Devuelve una lista de fragmentos, cada uno con su `chunk_id` y los
    metadatos mínimos exigidos: doc_id, version, pagina."""
    splitter = _construir_splitter(chunk_size, chunk_overlap)
    fragmentos = []

    for registro in registros_pagina:
        texto = registro["texto"]
        if not texto.strip():
            continue
        piezas = splitter.split_text(texto)
        for i, pieza in enumerate(piezas):
            fragmentos.append(
                {
                    "chunk_id": f"{registro['doc_id']}::p{registro['pagina']:04d}::c{i:03d}",
                    "doc_id": registro["doc_id"],
                    "titulo": registro["titulo"],
                    "institucion": registro["institucion"],
                    "tipo": registro["tipo"],
                    "version": registro.get("version", "sin_registrar"),
                    "pagina": registro["pagina"],
                    "indice_en_pagina": i,
                    "texto": pieza,
                    "n_tokens": contar_tokens(pieza),
                }
            )
    return fragmentos

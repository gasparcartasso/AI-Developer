
from __future__ import annotations

import hashlib
import re
from typing import Callable


# Normaliza el texto eliminando espacios repetidos.
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# Genera un ID estable para identificar cada chunk
# de forma reproducible.
def stable_chunk_id(
    document_id: str,
    strategy: str,
    index: int,
    text: str,
) -> str:

    # Genera un hash corto a partir del contenido del chunk.
    digest = hashlib.sha1(
        normalize_text(text).encode("utf-8")
    ).hexdigest()[:8]

    return f"{document_id}:{strategy}:{index:03d}:{digest}"


# Chunking de tamaño fijo:
# corta el texto por cantidad de palabras y mantiene overlap entre chunks.
def fixed_size_chunks(
    text: str,
    *,
    target_words: int = 55,
    overlap_words: int = 10,
) -> list[str]:

    # Valida el tamaño del chunk.
    if target_words <= 0:
        raise ValueError("target_words debe ser positivo")

    # El overlap debe ser menor que el tamaño del chunk.
    if not 0 <= overlap_words < target_words:
        raise ValueError("overlap_words inválido")

    # Normaliza y separa el texto en palabras.
    words = normalize_text(text).split()

    if not words:
        return []

    # Define cuánto avanza cada nuevo chunk.
    step = target_words - overlap_words

    result = []

    for start in range(0, len(words), step):
        piece = words[start:start + target_words]

        # Evita dejar un último chunk demasiado corto.
        if len(piece) < 10 and result:
            result[-1] = normalize_text(
                result[-1] + " " + " ".join(piece)
            )
            break

        result.append(" ".join(piece))

    return result


# Chunking recursivo:
# intenta respetar párrafos y oraciones antes de cortar por tamaño.
def recursive_chunks(
    text: str,
    *,
    target_words: int = 55,
    overlap_words: int = 8,
) -> list[str]:

    # Primero intenta separar el texto por párrafos.
    paragraphs = [
        normalize_text(part)
        for part in re.split(r"\n\s*\n", text)
        if normalize_text(part)
    ]

    if not paragraphs:
        paragraphs = [normalize_text(text)]

    units = []

    # Divide cada párrafo en oraciones.
    for paragraph in paragraphs:
        sentences = [
            normalize_text(sentence)
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                paragraph,
            )
            if normalize_text(sentence)
        ]

        units.extend(sentences or [paragraph])

    chunks = []
    current = []

    for unit in units:
        candidate = " ".join(current + [unit]).strip()

        # Si supera el tamaño objetivo, cierra el chunk actual.
        if current and len(candidate.split()) > target_words:
            chunks.append(" ".join(current))

            # Conserva algunas palabras del final como overlap.
            overlap = " ".join(
                " ".join(current).split()[-overlap_words:]
            )

            current = [overlap, unit] if overlap else [unit]

        else:
            current.append(unit)

    # Agrega el último chunk pendiente.
    if current:
        chunks.append(" ".join(current))

    return [
        normalize_text(chunk)
        for chunk in chunks
        if chunk.strip()
    ]


# Chunking semántico:
# agrupa oraciones mientras sus embeddings sean suficientemente similares.
def semantic_chunks(
    text: str,
    *,
    embed: Callable[[str], list[float]],
    similarity: Callable[[list[float], list[float]], float],
    threshold: float = 0.15,
    max_words: int = 70,
) -> list[str]:

    # Divide el texto en oraciones.
    sentences = [
        normalize_text(sentence)
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            normalize_text(text),
        )
        if normalize_text(sentence)
    ]

    if not sentences:
        return []

    # Comienza el primer chunk con la primera oración.
    chunks = [[sentences[0]]]

    # Genera el embedding de la primera oración.
    current_vector = embed(sentences[0])

    for sentence in sentences[1:]:
        candidate_words = len(
            " ".join(chunks[-1] + [sentence]).split()
        )

        # Genera el embedding de la nueva oración.
        sentence_vector = embed(sentence)

        # Comprueba si la nueva oración es suficientemente similar
        # al contenido actual del chunk.
        related = (
            similarity(current_vector, sentence_vector)
            >= threshold
        )

        # Si mantiene el tema y no supera el máximo,
        # agrega la oración al chunk actual.
        if related and candidate_words <= max_words:
            chunks[-1].append(sentence)

            # Recalcula el embedding del chunk completo.
            current_vector = embed(
                " ".join(chunks[-1])
            )

        else:
            # Si cambia el tema, comienza un nuevo chunk.
            chunks.append([sentence])
            current_vector = sentence_vector

    return [
        " ".join(group)
        for group in chunks
    ]


# Construye los chunks finales agregando ID, fuente y metadata
# para mantener trazabilidad con el documento original.
def build_chunks(
    documents: list[dict],
    *,
    strategy: str,
    splitter,
) -> list[dict]:

    items = []

    for document in documents:
        # Aplica la estrategia de chunking seleccionada.
        pieces = splitter(document["text"])

        for index, text in enumerate(pieces):
            items.append({
                # ID único y reproducible del chunk.
                "chunk_id": stable_chunk_id(
                    document["document_id"],
                    strategy,
                    index,
                    text,
                ),

                # Mantiene la relación con el documento original.
                "document_id": document["document_id"],

                # Contenido del chunk.
                "text": text,

                # Fuente original.
                "source": document["source"],

                # Página, si existe.
                "page": document.get("page"),

                # Metadata original + información del chunk.
                "metadata": {
                    **document.get("metadata", {}),
                    "chunk_strategy": strategy,
                    "chunk_index": index,
                },
            })

    return items


# Estrategia principal usada para documentos:
# aplica recursive chunking a todos los documentos recibidos.
def document_chunks(
    documents: list[dict],
    target_words: int = 55,
    overlap_words: int = 6,
) -> list[dict]:

    return build_chunks(
        documents,
        strategy="recursive",
        splitter=lambda text: recursive_chunks(
            text,
            target_words=target_words,
            overlap_words=overlap_words,
        ),
    )

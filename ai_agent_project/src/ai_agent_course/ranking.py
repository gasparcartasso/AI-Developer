from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Sequence

from .retrieval import SearchResult



def tokenize(text: str) -> list[str]:
    """Normaliza texto a tokens simples para el índice léxico.

    Args:
        text: Consulta o contenido de un chunk.

    Returns:
        Lista de tokens alfanuméricos en minúsculas.

    La misma normalización se aplica a documentos y queries para que BM25
    compare términos en un espacio léxico consistente.
    """
    return re.findall(r"\w+", text.lower())


@dataclass(frozen=True)
class LexicalResult:
    """Representa un chunk recuperado por BM25 junto con su score léxico."""

    chunk: dict
    score: float


@dataclass(frozen=True)
class FusedCandidate:
    """Candidato fusionado por RRF con trazabilidad de sus rankings de origen."""

    result: SearchResult
    rrf_score: float
    dense_rank: int | None
    lexical_rank: int | None


class BM25Retriever:
    """Retriever léxico basado en BM25 con fallback didáctico offline.

    Recibe chunks ya creados y devuelve coincidencias por términos. Su función
    arquitectónica es complementar al retriever dense con una señal distinta:
    nombres, siglas o palabras exactas que los embeddings pueden ordenar peor.
    """

    def __init__(
        self,
        chunks: Sequence[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Construye el índice léxico sobre los chunks del corpus.

        Args:
            chunks: Chunks con al menos ``text`` e identificadores.
            k1: Controla cuánto crece el aporte de repetir un término.
            b: Controla cuánto normaliza BM25 por longitud del documento.

        Returns:
            None. Deja el índice listo para usar con ``search``.
        """
        self.chunks = list(chunks)
        self.docs = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]
        self.k1 = k1
        self.b = b

        # Usamos la librería real cuando está disponible; si no, mantenemos
        # el mismo contrato con una implementación local para smoke mode.
        try:
            from rank_bm25 import BM25Okapi

            self._model = BM25Okapi(
                self.docs,
                k1=k1,
                b=b,
            )
            self.backend_name = "rank-bm25-okapi"
            self._fallback = None
        except ImportError:
            self._model = None
            self.backend_name = "didactic-bm25-fallback"
            self._fallback = self._build_fallback()

    def _build_fallback(self) -> dict:
        """Precalcula estadísticas necesarias para el BM25 local.

        No recibe parámetros externos: usa los documentos ya guardados en
        ``self.docs`` durante la construcción del retriever.

        Returns:
            Diccionario con IDF por término y longitud media del corpus.

        Este fallback permite estudiar la mecánica sin depender de rank-bm25;
        no pretende reemplazar una implementación optimizada de producción.
        """
        document_frequency = Counter()

        # Document frequency cuenta en cuántos documentos aparece cada término.
        # Usamos set(doc) para no contar repeticiones dentro del mismo documento.
        for doc in self.docs:
            document_frequency.update(set(doc))

        document_count = max(1, len(self.docs))
        average_length = (
            sum(len(doc) for doc in self.docs)
            / document_count
        )

        idf = {
            term: math.log(
                1
                + (
                    document_count
                    - frequency
                    + 0.5
                )
                / (
                    frequency
                    + 0.5
                )
            )
            for term, frequency in document_frequency.items()
        }

        return {
            "idf": idf,
            "avgdl": average_length,
        }

    def _fallback_scores(
        self,
        query_tokens: list[str],
    ) -> list[float]:
        """Calcula un score BM25 por documento para una consulta tokenizada.

        Args:
            query_tokens: Tokens de la consulta normalizados con ``tokenize``.

        Returns:
            Lista de scores alineada por posición con ``self.docs``.
        """
        scores = []

        for doc in self.docs:
            frequencies = Counter(doc)
            score = 0.0

            # Cada término aporta según frecuencia local (TF), rareza global
            # (IDF) y una corrección por longitud del documento.
            for term in query_tokens:
                frequency = frequencies.get(term, 0)

                if not frequency:
                    continue

                denominator = (
                    frequency
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b
                        * len(doc)
                        / max(self._fallback["avgdl"], 1e-9)
                    )
                )

                score += (
                    self._fallback["idf"].get(term, 0.0)
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                )

            scores.append(float(score))

        return scores

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[LexicalResult]:
        """Recupera los chunks con mayor score BM25.

        Args:
            query: Consulta en lenguaje natural.
            top_k: Cantidad máxima de candidatos a devolver.

        Returns:
            ``LexicalResult`` ordenados de mayor a menor score.

        Esta salida todavía no se mezcla con Dense: la fusión ocurre en RRF.
        """
        query_tokens = tokenize(query)

        if self._model is not None:
            scores = self._model.get_scores(query_tokens)
        else:
            scores = self._fallback_scores(query_tokens)

        rows = [
            LexicalResult(
                chunk=chunk,
                score=float(score),
            )
            for chunk, score in zip(
                self.chunks,
                scores,
                strict=True,
            )
        ]

        return sorted(
            rows,
            key=lambda row: row.score,
            reverse=True,
        )[:top_k]



def _as_search_result(chunk: dict) -> SearchResult:
    """Adapta un chunk léxico al contrato común ``SearchResult``.

    Args:
        chunk: Chunk original recuperado por BM25.

    Returns:
        ``SearchResult`` con la misma identidad y metadata del chunk.

    RRF necesita un payload común para candidatos que pueden provenir de Dense,
    de BM25 o de ambos.
    """
    return SearchResult(
        chunk_id=chunk["chunk_id"],
        document_id=chunk["document_id"],
        text=chunk["text"],
        score=0.0,
        source=chunk["source"],
        page=chunk.get("page"),
        metadata=chunk.get("metadata", {}),
    )



def rrf_fuse(
    dense_results: Sequence[SearchResult],
    lexical_results: Sequence[LexicalResult],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[FusedCandidate]:
    """Fusiona Dense y BM25 con Reciprocal Rank Fusion (RRF).

    Args:
        dense_results: Ranking producido por búsqueda vectorial.
        lexical_results: Ranking producido por BM25.
        k: Constante de RRF que suaviza la ventaja de las primeras posiciones.
        top_k: Máximo de candidatos fusionados a devolver.

    Returns:
        Candidatos ordenados por ``rrf_score`` con el rank de cada señal.

    RRF usa posiciones y no scores crudos porque Dense y BM25 viven en escalas
    diferentes; sumarlos directamente no tendría una interpretación estable.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, SearchResult] = {}
    dense_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}

    # Cada ranking aporta 1 / (k + rank). Un candidato presente en ambas
    # listas acumula evidencia de las dos señales.
    for rank, result in enumerate(dense_results, start=1):
        scores[result.chunk_id] = (
            scores.get(result.chunk_id, 0.0)
            + 1 / (k + rank)
        )
        payloads[result.chunk_id] = result
        dense_ranks[result.chunk_id] = rank

    for rank, lexical in enumerate(lexical_results, start=1):
        chunk = lexical.chunk
        chunk_id = chunk["chunk_id"]

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + 1 / (k + rank)
        )
        payloads.setdefault(
            chunk_id,
            _as_search_result(chunk),
        )
        lexical_ranks[chunk_id] = rank

    # Ordenamos una sola vez después de acumular las contribuciones de ambos rankings.
    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return [
        FusedCandidate(
            result=payloads[chunk_id],
            rrf_score=float(score),
            dense_rank=dense_ranks.get(chunk_id),
            lexical_rank=lexical_ranks.get(chunk_id),
        )
        for chunk_id, score in ordered
    ]

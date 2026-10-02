from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .ranking import FusedCandidate
from .retrieval import SearchResult


@dataclass(frozen=True)
class RankedCandidate:
    """Resultado reordenado que conserva el score RRF para trazabilidad."""

    result: SearchResult
    rerank_score: float
    rrf_score: float


class CrossEncoderReranker:
    """Reranker que puntúa ``query + texto`` conjuntamente con un Cross-Encoder.

    A diferencia de embeddings bi-encoder, el modelo observa ambos textos a la
    vez. Eso suele mejorar el orden fino, pero agrega costo por cada candidato.
    """

    default_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = default_model,
    ):
        """Carga el Cross-Encoder que se usará para reordenar candidatos.

        Args:
            model_name: Identificador del modelo compatible con sentence-transformers.

        Returns:
            None. Deja el modelo disponible en ``self.model``.
        """
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(
        self,
        query: str,
        candidates: Sequence[FusedCandidate],
        top_k: int = 3,
    ) -> list[RankedCandidate]:
        """Reordena un candidate set con un score de relevancia query-documento.

        Args:
            query: Consulta original del usuario.
            candidates: Candidatos ya recuperados y fusionados por RRF.
            top_k: Cantidad final de resultados que se conservará.

        Returns:
            ``RankedCandidate`` ordenados por score del Cross-Encoder.
        """
        # Cada inferencia compara la consulta con un candidato. Por eso conviene
        # rerankear pocos elementos y no usar este modelo sobre todo el corpus.
        pairs = [
            (query, candidate.result.text)
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        rows = [
            RankedCandidate(
                result=candidate.result,
                rerank_score=float(score),
                rrf_score=float(candidate.rrf_score),
            )
            for candidate, score in zip(
                candidates,
                scores,
                strict=True,
            )
        ]

        return sorted(
            rows,
            key=lambda row: row.rerank_score,
            reverse=True,
        )[:top_k]


class LexicalSmokeReranker:
    """Fallback offline que conserva el contrato sin simular semántica real."""

    name = "lexical-overlap-smoke-only"

    def rerank(
        self,
        query: str,
        candidates: Sequence[FusedCandidate],
        top_k: int = 3,
    ) -> list[RankedCandidate]:
        """Reordena candidatos con una señal léxica simple para smoke mode.

        Args:
            query: Consulta original.
            candidates: Candidate set producido por RRF.
            top_k: Cantidad final de elementos a devolver.

        Returns:
            ``RankedCandidate`` compatibles con la salida del Cross-Encoder.
        """
        query_tokens = set(
            re.findall(r"\w+", query.lower())
        )

        rows = []

        for candidate in candidates:
            candidate_tokens = set(
                re.findall(
                    r"\w+",
                    candidate.result.text.lower(),
                )
            )

            overlap = len(
                query_tokens & candidate_tokens
            )

            rows.append(
                RankedCandidate(
                    result=candidate.result,
                    rerank_score=float(overlap),
                    rrf_score=float(candidate.rrf_score),
                )
            )

        return sorted(
            rows,
            key=lambda row: row.rerank_score,
            reverse=True,
        )[:top_k]

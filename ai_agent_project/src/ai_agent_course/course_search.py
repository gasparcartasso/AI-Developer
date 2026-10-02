from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .ranking import BM25Retriever, rrf_fuse


SearchStrategy = Literal[
    "dense",
    "hybrid_rrf",
    "hybrid_rerank",
]


@dataclass
class SearchTrace:
    """Traza de diagnóstico que registra cuántos candidatos pasó cada etapa.

    No cambia el ranking: existe para observabilidad y para explicar qué camino
    siguió una búsqueda sin acoplar esa información al resultado final.
    """

    strategy: SearchStrategy
    dense_candidates: int
    lexical_candidates: int
    fused_candidates: int
    reranked_candidates: int
    reranker: str | None


class CourseSearchService:
    """Orquesta Dense, BM25, RRF y reranking detrás de una interfaz estable.

    La aplicación llama siempre a ``search``. La estrategia puede cambiar sin
    obligar al consumidor a conocer cómo se implementa cada etapa interna.
    """

    def __init__(
        self,
        dense_retriever,
        chunks,
        reranker,
        *,
        strategy: SearchStrategy = "hybrid_rerank",
        candidate_k: int = 8,
    ):
        """Configura los componentes internos del servicio de búsqueda.

        Args:
            dense_retriever: Retriever vectorial ya indexado.
            chunks: Corpus usado para construir el retriever BM25.
            reranker: Componente con método ``rerank`` y atributo ``name``.
            strategy: Estrategia usada por defecto.
            candidate_k: Tamaño del conjunto intermedio antes del top-k final.

        Returns:
            None. Deja listo el servicio para responder ``search``.
        """
        self.dense_retriever = dense_retriever
        self.lexical_retriever = BM25Retriever(chunks)
        self.reranker = reranker
        self.strategy = strategy
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        strategy: SearchStrategy | None = None,
    ):
        """Ejecuta la estrategia elegida manteniendo un contrato de salida común.

        Args:
            query: Consulta de búsqueda.
            top_k: Cantidad final de resultados.
            strategy: Override opcional de la estrategia configurada.

        Returns:
            Tupla ``(results, trace)``. ``results`` contiene SearchResult y
            ``trace`` describe cuántos candidatos recorrieron cada etapa.
        """
        chosen = strategy or self.strategy

        # Dense se ejecuta siempre: es el baseline y también una de las dos
        # señales que alimentan las variantes híbridas.
        dense = self.dense_retriever.search(
            query,
            top_k=max(self.candidate_k, top_k),
        )

        if chosen == "dense":
            return dense[:top_k], SearchTrace(
                strategy=chosen,
                dense_candidates=len(dense),
                lexical_candidates=0,
                fused_candidates=0,
                reranked_candidates=0,
                reranker=None,
            )

        # Solo las estrategias híbridas agregan la señal léxica de BM25.
        lexical = self.lexical_retriever.search(
            query,
            top_k=max(self.candidate_k, top_k),
        )

        # RRF construye un candidate set común antes de decidir si termina
        # aquí o si pasa por una etapa adicional de reranking.
        fused = rrf_fuse(
            dense,
            lexical,
            top_k=max(self.candidate_k, top_k),
        )

        if chosen == "hybrid_rrf":
            return (
                [candidate.result for candidate in fused[:top_k]],
                SearchTrace(
                    strategy=chosen,
                    dense_candidates=len(dense),
                    lexical_candidates=len(lexical),
                    fused_candidates=len(fused),
                    reranked_candidates=0,
                    reranker=None,
                ),
            )

        if chosen != "hybrid_rerank":
            raise ValueError(f"Estrategia desconocida: {chosen}")

        # El reranker opera sobre pocos candidatos para mejorar el orden fino
        # sin pagar el costo de un Cross-Encoder sobre todo el corpus.
        reranked = self.reranker.rerank(
            query,
            fused,
            top_k=top_k,
        )

        return (
            [candidate.result for candidate in reranked],
            SearchTrace(
                strategy=chosen,
                dense_candidates=len(dense),
                lexical_candidates=len(lexical),
                fused_candidates=len(fused),
                reranked_candidates=len(reranked),
                reranker=self.reranker.name,
            ),
        )

# Tests de contratos de retrieval que deben sobrevivir a futuros refactors.
from ai_agent_course.embeddings import DeterministicSemanticEmbeddings
from ai_agent_course.retrieval import InMemoryRetriever
from ai_agent_course.retrieval_eval import (
    hit_rate_at_k,
    precision_at_k,
    reciprocal_rank,
)


def build_test_retriever(chunks):
    # Usamos un provider determinístico para que los tests no dependan de APIs o descargas.
    return InMemoryRetriever(
        chunks,
        embedding_provider=DeterministicSemanticEmbeddings(),
    )


def test_filter_and_traceability():
    # Un filtro debe limitar resultados sin perder source ni document_id.
    chunks = [
        {
            "chunk_id": "A:0",
            "document_id": "A",
            "text": "Vacaciones con anticipación.",
            "source": "a.md",
            "page": 1,
            "metadata": {"department": "RRHH", "active": True, "version": "1"},
        },
        {
            "chunk_id": "B:0",
            "document_id": "B",
            "text": "Política de seguridad.",
            "source": "b.md",
            "page": 1,
            "metadata": {"department": "Tecnología", "active": True, "version": "1"},
        },
    ]

    retriever = build_test_retriever(chunks)

    results = retriever.search(
        "licencia anual",
        top_k=2,
        filters={"department": "RRHH"},
    )

    assert [result.document_id for result in results] == ["A"]
    assert results[0].source == "a.md"


def test_metrics_use_unique_documents():
    # Dos chunks del mismo documento deben ocupar una sola posición lógica.
    class Result:
        def __init__(self, document_id):
            self.document_id = document_id

    results = [
        Result("A"),
        Result("A"),
        Result("B"),
    ]

    # Ranking lógico: A primero, B segundo.
    assert hit_rate_at_k(results, ["B"], 2) == 1.0
    assert precision_at_k(results, ["B"], 2) == 0.5
    assert reciprocal_rank(results, ["B"]) == 0.5


def test_delete_document_removes_its_chunks():
    # Eliminar un document_id debe retirar todos sus chunks.
    chunks = [
        {
            "chunk_id": "A:0",
            "document_id": "A",
            "text": "Contenido que será eliminado.",
            "source": "a.md",
            "page": 1,
            "metadata": {"department": "QA", "active": True, "version": "1"},
        },
        {
            "chunk_id": "B:0",
            "document_id": "B",
            "text": "Contenido que debe permanecer.",
            "source": "b.md",
            "page": 1,
            "metadata": {"department": "QA", "active": True, "version": "1"},
        },
    ]

    retriever = build_test_retriever(chunks)
    count_before = retriever.count()

    retriever.delete_document("A")

    assert retriever.count() < count_before

    remaining = retriever.search(
        "contenido",
        top_k=10,
    )

    assert all(
        result.document_id != "A"
        for result in remaining
    )


def test_update_replaces_previous_version():
    # Actualizar = borrar la versión anterior y luego indexar la nueva.
    old_chunk = {
        "chunk_id": "A:0:old",
        "document_id": "A",
        "text": "La política anterior no requería validación.",
        "source": "a.md",
        "page": 1,
        "metadata": {"department": "RRHH", "active": True, "version": "1"},
    }

    new_chunk = {
        "chunk_id": "A:0:new",
        "document_id": "A",
        "text": "La nueva política requiere validación de RRHH.",
        "source": "a.md",
        "page": 1,
        "metadata": {"department": "RRHH", "active": True, "version": "2"},
    }

    retriever = build_test_retriever([old_chunk])

    retriever.delete_document("A")
    retriever.index([new_chunk], batch_size=1)

    results = retriever.search(
        "validación de RRHH",
        top_k=5,
    )

    assert any(
        result.document_id == "A"
        and result.metadata.get("version") == "2"
        for result in results
    )

    assert all(
        result.metadata.get("version") != "1"
        for result in results
        if result.document_id == "A"
    )

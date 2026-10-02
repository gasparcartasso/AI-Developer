
from ai_agent_course.chunking import fixed_size_chunks
from ai_agent_course.embeddings import DeterministicSemanticEmbeddings
from ai_agent_course.retrieval import InMemoryRetriever


# Verifica que el overlap no pueda ser igual o mayor
# que el tamaño total del chunk.
def test_overlap_validation():
    try:
        fixed_size_chunks(
            "texto",
            target_words=10,
            overlap_words=10,
        )
    except ValueError:
        return

    raise AssertionError(
        "Debía rechazar overlap inválido"
    )


# Verifica el contrato básico del retriever:
# una consulta relacionada debe recuperar el documento esperado.
def test_stable_retrieval_contract():
    chunks = [{
        "chunk_id": "D1:recursive:000:abc",
        "document_id": "D1",
        "text": "La política de vacaciones requiere anticipación.",
        "source": "doc.md",
        "page": 1,
        "metadata": {
            "department": "RRHH",
        },
    }]

    retriever = InMemoryRetriever(
        chunks,
        embedding_provider=DeterministicSemanticEmbeddings(),
    )

    result = retriever.search(
        "licencia anual",
        top_k=1,
    )[0]

    assert result.document_id == "D1"
    assert result.source == "doc.md"


# Verifica que al reindexar un documento se eliminen
# los chunks anteriores y quede solo la versión nueva.
def test_reindex_replaces_stale_chunks_and_keeps_ids_unique():
    provider = DeterministicSemanticEmbeddings()

    retriever = InMemoryRetriever(
        embedding_provider=provider
    )

    first = [{
        "chunk_id": "D1:recursive:000:old",
        "document_id": "D1",
        "text": "versión anterior vacaciones",
        "source": "doc.md",
        "page": None,
        "metadata": {
            "department": "RRHH",
        },
    }]

    replacement = [{
        "chunk_id": "D1:recursive:000:new",
        "document_id": "D1",
        "text": "versión actualizada vacaciones",
        "source": "doc.md",
        "page": None,
        "metadata": {
            "department": "RRHH",
        },
    }]

    # Indexa la primera versión.
    retriever.index(first)

    # Reindexa el mismo documento con contenido actualizado.
    retriever.index(replacement)

    # Debe quedar solamente el chunk nuevo.
    assert retriever.count() == 1

    assert (
        retriever
        .search("vacaciones", top_k=1)[0]
        .chunk_id
        .endswith(":new")
    )


# Verifica filtros, validaciones de búsqueda
# y eliminación de documentos del índice.
def test_filters_delete_and_invalid_search_inputs():
    chunks = [{
        "chunk_id": "D1:recursive:000:abc",
        "document_id": "D1",
        "text": "vacaciones licencia anual",
        "source": "doc.md",
        "page": 1,
        "metadata": {
            "department": "RRHH",
        },
    }]

    retriever = InMemoryRetriever(
        chunks,
        embedding_provider=DeterministicSemanticEmbeddings(),
    )

    # El filtro correcto debe devolver resultados.
    assert retriever.search(
        "licencia",
        filters={"department": "RRHH"},
    )

    # Un filtro que no coincide no debe devolver resultados.
    assert retriever.search(
        "licencia",
        filters={"department": "Finanzas"},
    ) == []

    # Verifica que se rechacen consultas inválidas.
    for query, top_k in [
        ("", 1),
        ("licencia", 0),
    ]:
        try:
            retriever.search(
                query,
                top_k=top_k,
            )

        except ValueError:
            pass

        else:
            raise AssertionError(
                "La búsqueda debía rechazar el input inválido"
            )

    # Elimina el documento del índice.
    retriever.delete_document("D1")

    # El retriever debe quedar vacío.
    assert retriever.count() == 0

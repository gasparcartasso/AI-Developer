
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from .embeddings import EmbeddingProvider, build_embedding_provider


# Representa un resultado de búsqueda del retriever.
@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str
    page: int | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


# Calcula similitud coseno entre dos vectores.
def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominator = float(
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    return (
        0.0
        if denominator == 0
        else float(np.dot(a, b) / denominator)
    )


# Retriever simple que mantiene chunks y vectores en memoria.
# Sirve como fallback didáctico sin base vectorial externa.
class InMemoryRetriever:
    backend_name = "in-memory-compatible"

    def __init__(
        self,
        chunks: Sequence[dict] | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        # Usa el provider recibido o el fallback determinístico.
        self.embedding_provider = (
            embedding_provider
            or build_embedding_provider(False)
        )

        # Acá se almacenan chunks y vectores.
        self._items: list[dict] = []

        if chunks:
            self.index(chunks)


    # Indexa chunks generando y almacenando sus embeddings.
    def index(
        self,
        chunks: Sequence[dict],
        batch_size: int = 32,
    ) -> None:

        chunks = list(chunks)

        if not chunks:
            return

        # Reemplaza chunks previos del mismo documento
        # para evitar versiones obsoletas.
        document_ids = {
            item["document_id"]
            for item in chunks
        }

        self._items = [
            item
            for item in self._items
            if item["document_id"] not in document_ids
        ]

        # Genera un embedding por cada chunk.
        vectors = self.embedding_provider.embed_documents(
            [item["text"] for item in chunks]
        )

        # Mantiene los chunks existentes por ID.
        by_id = {
            item["chunk_id"]: item
            for item in self._items
        }

        # Agrega cada chunk junto con su vector.
        for item, vector in zip(
            chunks,
            vectors,
            strict=True,
        ):
            by_id[item["chunk_id"]] = {
                **item,
                "vector": vector,
            }

        self._items = list(by_id.values())


    # Busca los chunks más similares a una consulta.
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:

        if not query.strip():
            raise ValueError(
                "La consulta no puede estar vacía"
            )

        if top_k < 1:
            raise ValueError(
                "top_k debe ser al menos 1"
            )

        # Convierte la consulta en un vector.
        query_vector = (
            self.embedding_provider.embed_query(query)
        )

        candidates = []

        for item in self._items:
            metadata = item.get("metadata", {})

            # Aplica filtros por metadata si existen.
            if filters and any(
                metadata.get(key) != value
                for key, value in filters.items()
            ):
                continue

            # Compara query y chunk con similitud coseno.
            score = _cosine(
                query_vector,
                item["vector"],
            )

            # Descarta resultados por debajo del umbral.
            if (
                min_score is not None
                and score < min_score
            ):
                continue

            candidates.append(
                (score, item)
            )

        # Ordena de mayor a menor similitud.
        candidates.sort(
            key=lambda pair: pair[0],
            reverse=True,
        )

        # Devuelve solo los top_k resultados.
        return [
            SearchResult(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                text=item["text"],
                score=float(score),
                source=item["source"],
                page=item.get("page"),
                metadata=dict(
                    item.get("metadata", {})
                ),
            )
            for score, item
            in candidates[:top_k]
        ]


    # Elimina todos los chunks de un documento.
    def delete_document(
        self,
        document_id: str,
    ) -> None:

        self._items = [
            item
            for item in self._items
            if item["document_id"] != document_id
        ]


    # En memoria no necesitamos índices adicionales.
    # Se mantiene el método para conservar la misma interfaz que Qdrant.
    def create_payload_indexes(
        self,
        fields: Sequence[str],
    ) -> None:
        return None


    # Devuelve la cantidad de chunks indexados.
    def count(self) -> int:
        return len(self._items)


    # Devuelve información básica del retriever.
    def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "points": self.count(),
            "embedding_model":
                self.embedding_provider.model_name,
            "vector_size":
                self.embedding_provider.dimension,
            "persistent": False,
        }


    # Exporta los chunks sin incluir sus vectores.
    def export_chunks(self) -> list[dict]:
        return [
            {
                key: value
                for key, value in item.items()
                if key != "vector"
            }
            for item in self._items
        ]


    def close(self) -> None:
        return None


# Convierte el chunk_id en un UUID reproducible para Qdrant.
def _point_id(chunk_id: str) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"ai-agent-course:{chunk_id}",
        )
    )


# Retriever persistente basado en Qdrant.
# Mantiene vectores, metadata e índice entre ejecuciones.
class QdrantRetriever:
    backend_name = "qdrant-local"

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "course_documents_v2",
        index_version: str = "class05-v1",
    ):

        try:
            from qdrant_client import (
                QdrantClient,
                models,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Instalá qdrant-client para usar Qdrant"
            ) from exc

        self._models = models
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self.index_version = index_version

        # Carpeta donde Qdrant guardará la información.
        self.path = Path(path)
        self.path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Guarda información sobre el índice y el modelo usado.
        self._manifest_path = (
            self.path
            / f"{collection_name}.manifest.json"
        )

        # Inicializa Qdrant local.
        self.client = QdrantClient(
            path=str(self.path)
        )

        # Crea o valida la colección.
        self._ensure_collection()


    # Define cómo debería haber sido creado el índice.
    def _expected_manifest(
        self,
    ) -> dict[str, Any]:

        return {
            "index_version": self.index_version,
            "embedding_model":
                self.embedding_provider.model_name,
            "embedding_dimensions":
                int(self.embedding_provider.dimension),
            "normalization": getattr(
                self.embedding_provider,
                "normalization",
                "unknown",
            ),
            "distance": "cosine",
        }


    # Verifica que el índice existente sea compatible
    # con el modelo de embeddings actual.
    def _validate_manifest(
        self,
        *,
        collection_created: bool,
    ) -> None:

        expected = self._expected_manifest()

        # Si es una colección nueva, crea el manifiesto.
        if not self._manifest_path.exists():
            if not collection_created:
                raise RuntimeError(
                    "La collection existe sin manifiesto "
                    "de compatibilidad. Usá otro "
                    "collection_name o reconstruí el índice."
                )

            self._manifest_path.write_text(
                json.dumps(
                    expected,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return

        # Lee el manifiesto existente.
        try:
            current = json.loads(
                self._manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "El manifiesto del índice no es válido"
            ) from exc

        # Detecta diferencias entre índice y provider actual.
        mismatches = {
            key: {
                "expected": value,
                "current": current.get(key),
            }
            for key, value in expected.items()
            if current.get(key) != value
        }

        if mismatches:
            raise RuntimeError(
                "Collection incompatible "
                f"con el provider actual: {mismatches}"
            )


    # Crea la collection si no existe y valida su configuración.
    def _ensure_collection(self):
        models = self._models

        collection_created = (
            not self.client.collection_exists(
                self.collection_name
            )
        )

        if collection_created:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_provider.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        # Verifica dimensión y métrica de distancia.
        info = self.client.get_collection(
            self.collection_name
        )

        vectors = info.config.params.vectors

        size = int(
            getattr(vectors, "size", 0)
        )

        if size != int(
            self.embedding_provider.dimension
        ):
            raise RuntimeError(
                f"Dimensión incompatible: "
                f"collection={size}, "
                f"provider="
                f"{self.embedding_provider.dimension}"
            )

        distance = str(
            getattr(vectors, "distance", "")
        ).upper()

        if not distance.endswith("COSINE"):
            raise RuntimeError(
                f"Distancia incompatible: "
                f"collection={distance}, "
                "expected=COSINE"
            )

        self._validate_manifest(
            collection_created=collection_created
        )


    # Convierte filtros simples en filtros compatibles con Qdrant.
    def _filter(self, filters):
        if not filters:
            return None

        models = self._models

        return models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(
                        value=value
                    ),
                )
                for key, value
                in filters.items()
            ]
        )


    # Indexa chunks en Qdrant en batches.
    def index(
        self,
        chunks: Sequence[dict],
        batch_size: int = 32,
    ):
        models = self._models
        chunks = list(chunks)

        if not chunks:
            return

        # Elimina versiones previas del mismo documento.
        for document_id in sorted({
            item["document_id"]
            for item in chunks
        }):
            self.delete_document(document_id)

        # Procesa los chunks en lotes.
        for start in range(
            0,
            len(chunks),
            batch_size,
        ):
            batch = chunks[
                start:start + batch_size
            ]

            # Genera embeddings del batch.
            vectors = (
                self.embedding_provider
                .embed_documents(
                    [
                        item["text"]
                        for item in batch
                    ]
                )
            )

            points = []

            for item, vector in zip(
                batch,
                vectors,
                strict=True,
            ):
                # Payload: datos asociados al vector.
                payload = {
                    "chunk_id": item["chunk_id"],
                    "document_id":
                        item["document_id"],
                    "text": item["text"],
                    "source": item["source"],
                    "page": item.get("page"),
                    **item.get("metadata", {}),
                    "embedding_model":
                        self.embedding_provider.model_name,
                }

                points.append(
                    models.PointStruct(
                        id=_point_id(
                            item["chunk_id"]
                        ),
                        vector=vector,
                        payload=payload,
                    )
                )

            # Inserta o actualiza los puntos en Qdrant.
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )


    # Busca los vectores más similares en Qdrant.
    def search(
        self,
        query,
        top_k=5,
        filters=None,
        min_score=None,
    ):

        if not str(query).strip():
            raise ValueError(
                "La consulta no puede estar vacía"
            )

        if top_k < 1:
            raise ValueError(
                "top_k debe ser al menos 1"
            )

        # Genera el embedding de la consulta.
        vector = (
            self.embedding_provider
            .embed_query(query)
        )

        # Qdrant realiza la búsqueda vectorial.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=self._filter(filters),
            limit=top_k,
            score_threshold=min_score,
            with_payload=True,
            with_vectors=False,
        )

        # Campos internos que no se devuelven como metadata.
        reserved = {
            "chunk_id",
            "document_id",
            "text",
            "source",
            "page",
            "embedding_model",
        }

        return [
            SearchResult(
                chunk_id=str(
                    point.payload["chunk_id"]
                ),
                document_id=str(
                    point.payload["document_id"]
                ),
                text=str(
                    point.payload["text"]
                ),
                score=float(point.score),
                source=str(
                    point.payload["source"]
                ),
                page=point.payload.get("page"),
                metadata={
                    key: value
                    for key, value
                    in point.payload.items()
                    if key not in reserved
                },
            )
            for point in response.points
        ]


    # Elimina todos los puntos asociados a un documento.
    def delete_document(
        self,
        document_id,
    ):
        models = self._models

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(
                                value=document_id
                            ),
                        )
                    ]
                )
            ),
            wait=True,
        )


    # Crea índices sobre campos de metadata
    # para acelerar búsquedas filtradas.
    def create_payload_indexes(
        self,
        fields,
    ):
        models = self._models

        for field in fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=
                    models.PayloadSchemaType.KEYWORD,
                wait=True,
            )


    # Devuelve la cantidad de puntos almacenados.
    def count(self):
        return int(
            self.client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        )


    # Devuelve información sobre el estado del índice.
    def health(self):
        return {
            "backend": self.backend_name,
            "collection": self.collection_name,
            "points": self.count(),
            "embedding_model":
                self.embedding_provider.model_name,
            "vector_size":
                self.embedding_provider.dimension,
            "normalization": getattr(
                self.embedding_provider,
                "normalization",
                "unknown",
            ),
            "index_version":
                self.index_version,
            "distance": "cosine",
            "persistent": True,
        }


    # Cierra la conexión con Qdrant.
    def close(self):
        self.client.close()


# Factory que permite elegir entre Qdrant
# y el retriever simple en memoria.
def build_retriever(
    *,
    use_qdrant: bool,
    path: str | Path,
    embedding_provider: EmbeddingProvider,
):
    if use_qdrant:
        return QdrantRetriever(
            path,
            embedding_provider=embedding_provider,
        )

    return InMemoryRetriever(
        embedding_provider=embedding_provider,
    )

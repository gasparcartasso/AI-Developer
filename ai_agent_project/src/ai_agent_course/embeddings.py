
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Protocol, Sequence

import numpy as np


class EmbeddingProvider(Protocol):
    # Contrato común para cualquier proveedor de embeddings.
    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _normalize(text: str) -> list[str]:
    # Pasa a minúsculas y separa letras de sus acentos.
    text = unicodedata.normalize("NFKD", text.lower())

    # Elimina los acentos para normalizar palabras equivalentes.
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Extrae las palabras del texto.
    tokens = re.findall(r"[a-z0-9]+", text)

    # Agrupa palabras relacionadas bajo un mismo término.
    synonyms = {
        "licencia": "vacaciones",
        "descanso": "vacaciones",
        "anual": "vacaciones",
        "reintegro": "gastos",
        "reembolsos": "gastos",
        "comprobante": "gastos",
        "credencial": "seguridad",
        "credenciales": "seguridad",
        "contrasena": "seguridad",
        "clave": "seguridad",
        "password": "seguridad",
        "home": "remoto",
        "office": "remoto",
        "teletrabajo": "remoto",
        "distancia": "remoto",
    }

    # Reemplaza sinónimos y devuelve los tokens normalizados.
    return [synonyms.get(token, token) for token in tokens]


class DeterministicSemanticEmbeddings:
    # Fallback didáctico: simula cierta similitud semántica
    # sin utilizar un modelo real de embeddings.
    model_name = "deterministic-semantic-smoke-v3"

    # Todos los vectores tendrán 192 dimensiones.
    dimension = 192

    # Los vectores se normalizan con norma L2.
    normalization = "l2"

    # Las primeras dimensiones representan tópicos conocidos.
    # Tópicos manuales para simular similitud semántica en el fallback didáctico.
    TOPICS = {
        0: {"vacaciones", "anticipacion"},
        1: {"gastos", "fiscal", "finanzas", "reembolso"},
        2: {"seguridad", "incidente", "sesiones", "exposicion"},
        3: {"remoto", "presencialidad", "disponibilidad"},
        4: {"capacitacion", "curso", "programa", "formacion"},
        5: {"reunion", "reuniones", "agenda", "decisiones"},
    }

    def _embed(self, text: str) -> list[float]:
        # Crea inicialmente un vector de 192 ceros.
        vector = np.zeros(self.dimension, dtype=float)

        # Normaliza el texto y obtiene sus tokens.
        tokens = _normalize(text)
        token_set = set(tokens)

        # Asigna más peso a las dimensiones de tópicos detectados.
        for dimension, keywords in self.TOPICS.items():
            matches = len(token_set & keywords)

            if matches:
                vector[dimension] = 4.0 + matches

        # Usa palabras y pares de palabras consecutivas como features
        # para agregar más detalle lexical al vector.
        features = tokens + [
            f"{tokens[index]}_{tokens[index + 1]}"
            for index in range(max(0, len(tokens) - 1))
        ]

        # Reservamos las primeras dimensiones para los tópicos.
        usable_dimensions = self.dimension - 8

        for feature in features:
            # Convierte cada feature en una posición reproducible del vector.
            digest = hashlib.sha256(
                feature.encode("utf-8")
            ).digest()

            # Usa solo las dimensiones posteriores a las reservadas para tópicos. (reservamos las primeras 8 dimensiones para los tópicos)
            index = (
                8
                + int.from_bytes(digest[:4], "big")
                % usable_dimensions
            )

            # El hash también determina si suma o resta.
            sign = 1.0 if digest[4] % 2 == 0 else -1.0

            vector[index] += 0.35 * sign

        # Normaliza el vector para que tenga longitud 1.
        norm = float(np.linalg.norm(vector)) or 1.0

        return (vector / norm).tolist()

    # Genera un embedding por cada documento.
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    # Genera el embedding de una consulta.
    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbeddings:
    # Implementación que utiliza un modelo real de embeddings.
    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ):
        try:
            from sentence_transformers import SentenceTransformer

        except ImportError as exc:
            raise RuntimeError(
                "Instalá sentence-transformers "
                "o usá el fallback determinístico."
            ) from exc

        # Carga el modelo seleccionado.
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

        # Obtiene automáticamente la dimensión del modelo.
        self.dimension = int(
            self._model.get_sentence_embedding_dimension()
        )

        self.normalization = "l2"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        # Genera embeddings reales para todos los textos.
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        # Una query se procesa igual que un documento individual.
        return self.embed_documents([text])[0]


def build_embedding_provider(use_real: bool = False,) -> EmbeddingProvider:

    # Si use_real=True utiliza Sentence Transformers.
    if use_real:
        # Permite cambiar el modelo mediante una variable de entorno.
        model_name = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            (
                "sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2"
            ),
        )

        return SentenceTransformerEmbeddings(model_name)

    # Por defecto usa el fallback didáctico y determinístico.
    return DeterministicSemanticEmbeddings()

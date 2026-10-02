# Persiste la caché reutilizable que usará el provider durante la indexación.
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence


# El archivo permite reutilizar vectores incluso entre ejecuciones de la notebook.
class JsonEmbeddingCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {}
        )
        self.hits = 0
        self.misses = 0

    def key(self, model_name, text):
        payload = f"{model_name}\0{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, model_name, text):
        value = self.data.get(self.key(model_name, text))
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, model_name, text, vector):
        self.data[self.key(model_name, text)] = vector

    def flush(self):
        self.path.write_text(
            json.dumps(self.data),
            encoding="utf-8",
        )

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "entries": len(self.data),
        }


# Envuelve un provider de embeddings y consulta la caché antes de calcular.
class CachedEmbeddingProvider:
    def __init__(self, base, cache):
        self.base = base
        self.cache = cache
        self.model_name = base.model_name
        self.dimension = base.dimension

    def embed_documents(self, texts: Sequence[str]):
        texts = list(texts)

        # Conserva el orden original de los textos.
        output = [None] * len(texts)

        # Registramos qué posiciones necesitan cálculo real.
        missing_positions = []
        missing_texts = []

        for index, text in enumerate(texts):
            cached = self.cache.get(self.model_name, text)
            if cached is None:
                missing_positions.append(index)
                missing_texts.append(text)
            else:
                output[index] = cached

        # Calculamos EN BATCH únicamente los embeddings que no estaban cacheados.
        if missing_texts:
            vectors = self.base.embed_documents(missing_texts)
            for position, text, vector in zip(
                missing_positions, missing_texts, vectors
            ):
                output[position] = vector
                self.cache.set(self.model_name, text, vector)
            self.cache.flush()

        return list(output)

    def embed_query(self, text):
        # Las queries suelen ser dinámicas y poco repetitivas.
        # Priorizamos cachear documentos, que se reutilizan entre búsquedas.
        return self.base.embed_query(text)

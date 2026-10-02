from __future__ import annotations

import os
import re
from typing import Protocol, Sequence

from pydantic import BaseModel


class GenerationEvidence(BaseModel):
    source: str
    chunk_id: str
    text: str


class Generator(Protocol):
    """Contrato común para los backends de generación."""

    name: str

    def generate(
        self,
        *,
        question: str,
        context: str,
        prompt: str,
        evidence: Sequence[GenerationEvidence],
    ) -> str: ...


class GeminiGenerator:
    """Generador basado en Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY vacío")

        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.name = model

    def generate(
        self,
        *,
        question: str,
        context: str,
        prompt: str,
        evidence: Sequence[GenerationEvidence],
    ) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=450,
            ),
        )

        return (response.text or "").strip()


class ExtractiveOfflineGenerator:
    """Fallback offline que reutiliza frases de la evidencia recibida."""

    name = "extractive-offline-fallback"

    def generate(
        self,
        *,
        question: str,
        context: str,
        prompt: str,
        evidence: Sequence[GenerationEvidence],
    ) -> str:
        query_tokens = set(re.findall(r"\w+", question.lower()))
        ranked = []

        for item in evidence:
            for sentence in re.split(r"(?<=[.!?])\s+", item.text):
                sentence = sentence.strip()

                if not sentence:
                    continue

                sentence_tokens = set(
                    re.findall(r"\w+", sentence.lower())
                )

                overlap = len(query_tokens & sentence_tokens)

                ranked.append(
                    (overlap, sentence, item)
                )

        ranked.sort(
            key=lambda row: row[0],
            reverse=True,
        )

        useful = [
            row
            for row in ranked
            if row[0] > 0
        ]

        if not useful:
            return "NO_EVIDENCE"

        return " ".join(
            f"{sentence} [SOURCE: {item.source} | CHUNK: {item.chunk_id}]"
            for _, sentence, item in useful[:2]
        )


def build_generator() -> Generator:
    """Construye el backend de generación según la configuración."""

    use_real = os.getenv("USE_REAL_GEMINI", "0") == "1"
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if use_real:
        if not api_key:
            raise RuntimeError(
                "USE_REAL_GEMINI=1 requiere "
                "GEMINI_API_KEY o GOOGLE_API_KEY."
            )

        return GeminiGenerator(
            api_key=api_key,
            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.1-flash-lite",
            ),
        )

    return ExtractiveOfflineGenerator()

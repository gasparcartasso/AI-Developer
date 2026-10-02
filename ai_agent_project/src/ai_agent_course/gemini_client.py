from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    temperature: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


class GeminiClient:
    def __init__(self, env_path=None):
        if env_path is not None:
            load_dotenv(env_path)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta GEMINI_API_KEY")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Falta instalar google-genai. Ejecutá: python -m pip install google-genai"
            ) from exc

        self._types = types
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")
        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
            ),
        )
        usage = response.usage_metadata
        return GenerationResult(
            text=response.text or "",
            model=self.model,
            temperature=temperature,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            total_tokens=int(getattr(usage, "total_token_count", 0) or 0),
        )

    def count_tokens(self, text: str) -> int:
        if not text.strip():
            return 0
        counted = self.client.models.count_tokens(
            model=self.model,
            contents=text,
        )
        return int(counted.total_tokens)

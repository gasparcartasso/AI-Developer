from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import time
from typing import AsyncIterator, Protocol

from .errors import (
    InvalidProviderResponseError,
    InvalidRequestError,
    ProviderConfigurationError,
    ProviderTimeoutError,
    RateLimitError,
    TransientProviderError,
)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"

    def to_dict(self) -> dict:
        return asdict(self)


class LLMProvider(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> GenerationResult:
        ...

    async def agenerate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> GenerationResult:
        ...

    async def astream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        ...


class FakeProvider:
    """Provider determinístico para aprender y testear sin red ni costo."""

    def __init__(
        self,
        model: str = "fake-llm",
        *,
        delay_seconds: float = 0.03,
        failures: list[str] | None = None,
    ):
        self.model = model
        self.delay_seconds = delay_seconds
        self.failures = list(failures or [])
        self.active_calls = 0
        self.max_active_calls = 0

    def _validate(self, prompt: str) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidRequestError("El prompt no puede estar vacío")

    def _maybe_fail(self) -> None:
        if not self.failures:
            return
        failure = self.failures.pop(0)
        mapping = {
            "rate_limit": RateLimitError("429 simulado: demasiadas solicitudes"),
            "transient": TransientProviderError("503 simulado: proveedor no disponible"),
            "timeout": ProviderTimeoutError("timeout simulado por el provider"),
            "empty": InvalidProviderResponseError("respuesta simulada vacía"),
            "config": ProviderConfigurationError("credencial simulada inválida"),
        }
        raise mapping[failure]

    @staticmethod
    def _answer(prompt: str) -> str:
        marker = "NUEVO MENSAJE DEL USUARIO"
        if marker in prompt:
            tail = prompt.split(marker, 1)[1]
            user_text = tail.split("Respondé considerando el historial reciente.", 1)[0].strip()
            return f"Respuesta simulada para: {user_text[:120]}"
        last_line = next(
            (line.strip() for line in reversed(prompt.splitlines()) if line.strip()),
            prompt.strip(),
        )
        return f"Respuesta simulada para: {last_line[:120]}"

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> GenerationResult:
        self._validate(prompt)
        started = time.perf_counter()
        time.sleep(self.delay_seconds)
        self._maybe_fail()
        text = self._answer(prompt)
        return GenerationResult(
            text=text,
            model=self.model,
            provider="fake",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(text.split())),
        )

    async def agenerate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> GenerationResult:
        self._validate(prompt)
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        started = time.perf_counter()
        try:
            await asyncio.sleep(self.delay_seconds)
            self._maybe_fail()
            text = self._answer(prompt)
            return GenerationResult(
                text=text,
                model=self.model,
                provider="fake",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                input_tokens=max(1, len(prompt.split())),
                output_tokens=max(1, len(text.split())),
            )
        finally:
            self.active_calls -= 1

    async def astream(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> AsyncIterator[str]:
        self._validate(prompt)
        self._maybe_fail()
        for word in self._answer(prompt).split():
            await asyncio.sleep(self.delay_seconds / 4)
            yield word + " "


class GeminiProvider:
    """Adaptador opcional para Google Gen AI SDK."""

    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise ProviderConfigurationError("Falta GEMINI_API_KEY")
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _map_exception(exc: Exception) -> Exception:
        text = str(exc).lower()
        if "429" in text or "rate limit" in text or "resource exhausted" in text:
            return RateLimitError(str(exc))
        if "timeout" in text or "timed out" in text:
            return ProviderTimeoutError(str(exc))
        if "401" in text or "403" in text or "api key" in text:
            return ProviderConfigurationError(str(exc))
        return TransientProviderError(str(exc))

    @staticmethod
    def _config(system: str | None, temperature: float):
        from google.genai import types
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
        )

    @staticmethod
    def _usage(response) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None)
        return (
            int(getattr(usage, "prompt_token_count", 0) or 0),
            int(getattr(usage, "candidates_token_count", 0) or 0),
        )

    def _normalize(self, response, started: float) -> GenerationResult:
        text = response.text or ""
        if not text.strip():
            raise InvalidProviderResponseError("Gemini devolvió texto vacío")
        input_tokens, output_tokens = self._usage(response)
        return GenerationResult(
            text=text,
            model=self.model,
            provider="gemini",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> GenerationResult:
        if not prompt.strip():
            raise InvalidRequestError("El prompt no puede estar vacío")
        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config(system, temperature),
            )
            return self._normalize(response, started)
        except Exception as exc:
            if isinstance(exc, InvalidProviderResponseError):
                raise
            raise self._map_exception(exc) from exc

    async def agenerate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> GenerationResult:
        if not prompt.strip():
            raise InvalidRequestError("El prompt no puede estar vacío")
        started = time.perf_counter()
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config(system, temperature),
            )
            return self._normalize(response, started)
        except Exception as exc:
            if isinstance(exc, InvalidProviderResponseError):
                raise
            raise self._map_exception(exc) from exc

    async def astream(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> AsyncIterator[str]:
        if not prompt.strip():
            raise InvalidRequestError("El prompt no puede estar vacío")
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=self._config(system, temperature),
            )
            async for chunk in stream:
                text = chunk.text or ""
                if text:
                    yield text
        except Exception as exc:
            raise self._map_exception(exc) from exc


def build_provider(settings):
    if settings.use_real_gemini:
        return GeminiProvider(settings.api_key_value(), settings.default_model)
    return FakeProvider(model=f"fake:{settings.default_model}")

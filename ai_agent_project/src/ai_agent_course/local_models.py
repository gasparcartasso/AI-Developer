from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

import httpx

from .providers import GenerationResult


class LocalProviderError(RuntimeError):
    pass


class SimulatedAgentProvider:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        latency_ms: float,
        supports_tools: bool,
        structured_reliability: float,
    ):
        self.name = name
        self.model = model
        self.latency_ms = latency_ms
        self.supports_tools = supports_tools
        self.structured_reliability = structured_reliability

    def _response_for(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "exactamente 7 palabras" in lowered:
            return "Agente decide acciones usando herramientas y contexto"
        if "seleccioná una herramienta" in lowered:
            return (
                '{"tool":"buscar_politica","arguments":{"query":"vacaciones"}}'
                if self.supports_tools
                else "No puedo seleccionar herramientas de forma confiable."
            )
        if "solo json" in lowered:
            if self.structured_reliability < 0.90:
                return "category=rrhh; confidence=0.93"
            return '{"category":"rrhh","confidence":0.93}'
        if "resumí" in lowered:
            return "Resumen: la política requiere registro, revisión y aprobación."
        return f"Respuesta de {self.name}: {prompt[:100]}"

    async def agenerate(self, prompt: str, *, system=None, temperature=0.2):
        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")
        started = time.perf_counter()
        await asyncio.sleep(self.latency_ms / 1000)
        text = self._response_for(prompt)
        return GenerationResult(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(text.split())),
        )

    async def astream(self, prompt: str, *, system=None, temperature=0.2):
        result = await self.agenerate(prompt, system=system, temperature=temperature)
        for token in result.text.split():
            await asyncio.sleep(0.005)
            yield token + " "

    async def achat_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict],
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")
        if not tools:
            raise ValueError("Se requiere al menos una tool")

        started = time.perf_counter()
        await asyncio.sleep(self.latency_ms / 1000)
        tool_calls = []
        if self.supports_tools:
            function = tools[0].get("function", {})
            tool_calls = [{
                "function": {
                    "name": function.get("name", "buscar_politica"),
                    "arguments": {"query": "vacaciones"},
                }
            }]

        return {
            "content": "" if tool_calls else "No se solicitó una herramienta.",
            "model": self.model,
            "provider": self.name,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "tool_calls": tool_calls,
        }


class OllamaProvider:
    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
        fallback=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback

    async def agenerate(self, prompt: str, *, system=None, temperature=0.2):
        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature},
        }
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            if self.fallback is not None:
                return await self.fallback.agenerate(
                    prompt, system=system, temperature=temperature
                )
            raise LocalProviderError(f"Ollama no disponible: {exc}") from exc

        text = str(data.get("response", "")).strip()
        if not text:
            raise LocalProviderError("Ollama devolvió una respuesta vacía")

        return GenerationResult(
            text=text,
            model=self.model,
            provider="ollama",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
        )

    async def astream(self, prompt: str, *, system=None, temperature=0.2) -> AsyncIterator[str]:
        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/generate", json=payload
                ) as response:
                    response.raise_for_status()
                    emitted = False
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        token = str(data.get("response", ""))
                        if token:
                            emitted = True
                            yield token
                    if not emitted:
                        raise LocalProviderError("El stream no emitió contenido")
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            if self.fallback is None:
                raise LocalProviderError(f"Falló el streaming local: {exc}") from exc
            async for token in self.fallback.astream(
                prompt, system=system, temperature=temperature
            ):
                yield token

    async def achat_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict],
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Solicita tool calling nativo mediante Ollama /api/chat.

        Esta operación detecta la intención de llamar una herramienta, pero no la
        ejecuta. La ejecución y el loop de observación se trabajan más adelante.
        """
        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")
        if not tools:
            raise ValueError("Se requiere al menos una tool")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": temperature},
        }

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            if self.fallback is not None and hasattr(self.fallback, "achat_with_tools"):
                return await self.fallback.achat_with_tools(
                    prompt,
                    tools=tools,
                    system=system,
                    temperature=temperature,
                )
            raise LocalProviderError(f"Falló el tool calling local: {exc}") from exc

        message = data.get("message", {})
        return {
            "content": str(message.get("content", "")).strip(),
            "model": str(data.get("model", self.model)),
            "provider": "ollama",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "input_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(data.get("eval_count", 0) or 0),
            "tool_calls": message.get("tool_calls", []) or [],
        }

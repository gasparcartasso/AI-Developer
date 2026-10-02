from __future__ import annotations

from dataclasses import dataclass
import asyncio
import random
from typing import Callable

from .errors import ProviderTimeoutError, RateLimitError, TransientProviderError
from .providers import GenerationResult, LLMProvider


RETRYABLE_ERRORS = (RateLimitError, ProviderTimeoutError, TransientProviderError)


@dataclass(frozen=True)
class RetryEvent:
    attempt: int
    error_type: str
    message: str
    next_delay_seconds: float


@dataclass(frozen=True)
class CallOutcome:
    result: GenerationResult
    attempts: int
    retries: tuple[RetryEvent, ...]


async def generate_with_resilience(
    provider: LLMProvider,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    timeout_seconds: float = 20.0,
    max_retries: int = 2,
    base_delay_seconds: float = 0.25,
    jitter: bool = False,
    on_retry: Callable[[RetryEvent], None] | None = None,
) -> CallOutcome:
    retries: list[RetryEvent] = []

    for attempt in range(1, max_retries + 2):
        try:
            result = await asyncio.wait_for(
                provider.agenerate(prompt, system=system, temperature=temperature),
                timeout=timeout_seconds,
            )
            return CallOutcome(result=result, attempts=attempt, retries=tuple(retries))
        except asyncio.TimeoutError:
            error = ProviderTimeoutError(
                f"La llamada superó {timeout_seconds:.2f} segundos"
            )
        except RETRYABLE_ERRORS as exc:
            error = exc

        if attempt > max_retries:
            raise error

        delay = base_delay_seconds * (2 ** (attempt - 1))
        if jitter:
            delay *= random.uniform(0.8, 1.2)

        event = RetryEvent(
            attempt=attempt,
            error_type=type(error).__name__,
            message=str(error),
            next_delay_seconds=round(delay, 4),
        )
        retries.append(event)
        if on_retry:
            on_retry(event)
        await asyncio.sleep(delay)

    raise RuntimeError("Estado inalcanzable")

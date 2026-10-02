from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import uuid

from .conversation import ConversationSession
from .errors import PartialStreamError
from .resilience import RetryEvent, generate_with_resilience
from .runtime import JsonlTraceStore


@dataclass(frozen=True)
class ChatResponse:
    run_id: str
    session_id: str
    text: str
    attempts: int
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StreamChatResponse:
    run_id: str
    session_id: str
    text: str
    chunks: int
    first_chunk_ms: float
    total_ms: float
    provider: str
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


class ConversationApp:
    def __init__(self, provider, settings, trace_store: JsonlTraceStore):
        self.provider = provider
        self.settings = settings
        self.trace_store = trace_store

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            input_tokens / 1_000_000 * self.settings.input_usd_per_million
            + output_tokens / 1_000_000 * self.settings.output_usd_per_million,
            8,
        )

    def _start_run(self, *, run_id: str, session: ConversationSession, mode: str) -> None:
        self.trace_store.append(
            run_id=run_id,
            session_id=session.session_id,
            event="run.started",
            payload={
                "mode": mode,
                "provider": type(self.provider).__name__,
                "model": getattr(self.provider, "model", "unknown"),
                "history_messages": len(session.recent_messages()),
            },
        )

    async def ask(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        system: str = "Sos un asistente útil, preciso y breve.",
    ) -> ChatResponse:
        clean_text = user_text.strip()
        if not clean_text:
            raise ValueError("El mensaje no puede estar vacío")

        run_id = str(uuid.uuid4())
        prompt = session.build_prompt(clean_text)
        self._start_run(run_id=run_id, session=session, mode="async")

        def on_retry(event: RetryEvent) -> None:
            self.trace_store.append(
                run_id=run_id,
                session_id=session.session_id,
                event="llm.retry",
                payload=asdict(event),
            )

        try:
            outcome = await generate_with_resilience(
                self.provider,
                prompt,
                system=system,
                timeout_seconds=self.settings.request_timeout_seconds,
                max_retries=self.settings.max_retries,
                base_delay_seconds=self.settings.retry_base_delay_seconds,
                on_retry=on_retry,
            )
            result = outcome.result
            session.add("user", clean_text)
            session.add("assistant", result.text)

            response = ChatResponse(
                run_id=run_id,
                session_id=session.session_id,
                text=result.text,
                attempts=outcome.attempts,
                provider=result.provider,
                model=result.model,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                estimated_cost_usd=self._estimate_cost(
                    result.input_tokens,
                    result.output_tokens,
                ),
            )
            self.trace_store.append(
                run_id=run_id,
                session_id=session.session_id,
                event="run.completed",
                payload=response.to_dict(),
            )
            return response
        except Exception as exc:
            self.trace_store.append(
                run_id=run_id,
                session_id=session.session_id,
                event="run.failed",
                payload={"error_type": type(exc).__name__, "message": str(exc)},
            )
            raise

    async def ask_stream(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        system: str = "Sos un asistente útil, preciso y breve.",
    ) -> StreamChatResponse:
        clean_text = user_text.strip()
        if not clean_text:
            raise ValueError("El mensaje no puede estar vacío")

        run_id = str(uuid.uuid4())
        prompt = session.build_prompt(clean_text)
        self._start_run(run_id=run_id, session=session, mode="stream")

        started = time.perf_counter()
        chunks: list[str] = []
        first_chunk_ms: float | None = None

        try:
            async for chunk in self.provider.astream(prompt, system=system):
                chunks.append(chunk)
                if first_chunk_ms is None:
                    first_chunk_ms = round((time.perf_counter() - started) * 1000, 2)
        except Exception as exc:
            if chunks:
                error = PartialStreamError(
                    f"El stream falló después de {len(chunks)} fragmentos: {exc}"
                )
            else:
                error = exc
            self.trace_store.append(
                run_id=run_id,
                session_id=session.session_id,
                event="run.failed",
                payload={"error_type": type(error).__name__, "message": str(error)},
            )
            raise error

        text = "".join(chunks).strip()
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        session.add("user", clean_text)
        session.add("assistant", text)

        response = StreamChatResponse(
            run_id=run_id,
            session_id=session.session_id,
            text=text,
            chunks=len(chunks),
            first_chunk_ms=first_chunk_ms or total_ms,
            total_ms=total_ms,
            provider="gemini" if type(self.provider).__name__ == "GeminiProvider" else "fake",
            model=getattr(self.provider, "model", "unknown"),
        )
        self.trace_store.append(
            run_id=run_id,
            session_id=session.session_id,
            event="run.completed",
            payload=response.to_dict(),
        )
        return response

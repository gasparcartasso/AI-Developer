from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from contextlib import contextmanager
from typing import Any
from uuid import uuid4


SECRET_MARKERS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "credential",
    "private_key",
}


def utc_now() -> datetime:
    """Devuelve la hora actual en UTC con timezone explícito."""
    return datetime.now(timezone.utc)


def _iso(value: datetime | None,) -> str | None:
    """Serializa un datetime a ISO-8601 UTC; preserva ``None``."""
    if value is None:
        return None

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Usá datetime con timezone explícito")
    return value.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str | None,) -> datetime | None:
    """Reconstruye un datetime ISO-8601; devuelve ``None`` si no hay valor."""
    return (
        datetime.fromisoformat(value)
        if value
        else None
    )


def _tokens(text: str,) -> set[str]:
    """Tokeniza de forma léxica para el ranking didáctico de memoria."""
    return {
        token
        for token in re.findall(
            r"\w+",
            str(text).casefold(),
            flags=re.UNICODE,
        )
        if len(token) > 1
    }


@dataclass(frozen=True)
class MemoryScope:
    """Namespace de aislamiento para recuerdos persistentes.

    ``user_id`` es obligatorio. Proyecto y curso permiten separar contextos.
    ``session_id`` queda vacío en la demo de memoria durable para que el dato
    sobreviva a sesiones; puede usarse si deliberadamente queremos persistencia
    restringida a una sesión concreta.
    """

    user_id: str
    project_id: str = ""
    course_id: str = ""
    session_id: str = ""


@dataclass(frozen=True)
class MemoryCandidate:
    """Dato candidato junto con señales necesarias para el Write Gate."""

    kind: str
    key: str
    value: Any
    source: str
    confirmed: bool = False
    durable: bool = False
    sensitive: bool = False
    persist_allowed: bool = True
    confidence: float = 1.0
    priority: int = 5
    expires_at: datetime | None = None


@dataclass(frozen=True)
class MemoryWriteDecision:
    """Resultado explicable del Write Gate."""

    accepted: bool
    reason: str


def evaluate_write(candidate: MemoryCandidate,) -> MemoryWriteDecision:
    """Evalúa gobernanza antes de persistir un candidato.

    Rechaza datos no permitidos, sensibles, no confirmados, transitorios o de
    tipos que pertenecen al contexto de ejecución y no a memoria durable.
    """
    normalized_key = (candidate.key.casefold())

    if not candidate.persist_allowed:
        return MemoryWriteDecision(
            False,
            "persistencia no permitida",
        )

    if (
        candidate.sensitive
        or any(
            marker
            in normalized_key
            for marker
            in SECRET_MARKERS
        )
    ):
        return MemoryWriteDecision(
            False,
            "dato sensible",
        )

    if not candidate.confirmed:
        return MemoryWriteDecision(
            False,
            "dato no confirmado",
        )

    if not candidate.durable:
        return MemoryWriteDecision(
            False,
            "dato transitorio",
        )

    if candidate.kind in {
        "transient",
        "tool_result",
        "raw_evidence",
    }:
        return MemoryWriteDecision(
            False,
            (
                "tipo no apto para "
                "memoria estable"
            ),
        )

    return MemoryWriteDecision(
        True,
        "aceptado",
    )


class MemoryStore:
    """Store SQLite con aislamiento, provenance, TTL y lifecycle explícito."""

    def __init__(
        self,
        path: str | Path,
    ):
        """Crea/abre la base y asegura el schema mínimo de memoria."""
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    course_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    supersedes TEXT,
                    status TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_scope
                ON memory (
                    user_id,
                    project_id,
                    course_id,
                    session_id,
                    status
                )
                """
            )

    @contextmanager
    def _connection(self):
        """Abre una conexión, confirma/revierte la transacción y siempre cierra.

        El context manager de sqlite3 solo gestiona la transacción; no cierra
        la conexión. El finally libera el archivo también si ocurre un error.
        """
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def reset(self):
        """Vacía la base dedicada a la demo sin borrar el archivo SQLite.

        No recibe argumentos ni devuelve datos. Se usa únicamente al reiniciar
        esta práctica; elimina todos sus recuerdos, incluidos los históricos.
        """
        with self._connection() as conn:
            conn.execute("DELETE FROM memory")

    def _insert_active(
        self,
        conn: sqlite3.Connection,
        *,
        scope: MemoryScope,
        candidate: MemoryCandidate,
        supersedes: str | None,
        now: datetime,
    ) -> str:
        """Inserta un recuerdo activo usando una transacción ya abierta.

        Separar esta operación permite que ``supersede`` cambie estado e inserte
        la nueva versión de forma atómica, sin dejar el store a mitad de camino.
        """
        memory_id = (
            "mem_"
            + uuid4().hex[:12]
        )

        conn.execute(
            """
            INSERT INTO memory (
                memory_id,
                user_id,
                project_id,
                course_id,
                session_id,
                kind,
                key,
                value_json,
                source,
                confidence,
                priority,
                created_at,
                updated_at,
                expires_at,
                supersedes,
                status
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, 'active'
            )
            """,
            (
                memory_id,
                scope.user_id,
                scope.project_id,
                scope.course_id,
                scope.session_id,
                candidate.kind,
                candidate.key,
                json.dumps(
                    candidate.value,
                    ensure_ascii=False,
                ),
                candidate.source,
                float(candidate.confidence),
                int(candidate.priority),
                _iso(now),
                _iso(now),
                _iso(candidate.expires_at),
                supersedes,
            ),
        )

        return memory_id

    def write(
        self,
        *,
        scope: MemoryScope,
        candidate: MemoryCandidate,
        supersedes: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """Persiste un candidato que haya superado el Write Gate.

        Devuelve el ``memory_id`` generado. Si el candidato es rechazado, lanza
        ``ValueError`` y no modifica la base.
        """
        if supersedes is not None:
            # Supersede debe desactivar la versión anterior en la misma transacción.
            return self.supersede(supersedes, scope=scope, candidate=candidate, now=now)
        decision = evaluate_write(candidate)

        if not decision.accepted:
            raise ValueError(
                "Memory candidate rechazado: "
                + decision.reason
            )

        now = now or utc_now()

        with self._connection() as conn:
            return self._insert_active(
                conn,
                scope=scope,
                candidate=candidate,
                supersedes=supersedes,
                now=now,
            )

    def _row_to_dict(
        self,
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        """Convierte una fila SQLite y deserializa ``value_json``."""
        item = dict(row)

        item["value"] = json.loads(item.pop("value_json"))

        return item

    def list_scope(
        self,
        scope: MemoryScope,
        *,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        """Lista recuerdos dentro del scope exacto, opcionalmente históricos."""
        status_sql = (
            ""
            if include_inactive
            else "AND status='active'"
        )

        with self._connection() as conn:
            conn.row_factory = (sqlite3.Row)

            rows = conn.execute(
                f"""
                SELECT *
                FROM memory
                WHERE user_id=?
                  AND project_id=?
                  AND course_id=?
                  AND session_id=?
                  {status_sql}
                ORDER BY updated_at DESC
                """,
                (
                    scope.user_id,
                    scope.project_id,
                    scope.course_id,
                    scope.session_id,
                ),
            ).fetchall()

        return [
            self._row_to_dict(row)
            for row
            in rows
        ]

    def retrieve(
        self,
        *,
        scope: MemoryScope,
        query: str | None = None,
        kinds: set[str] | None = None,
        limit: int = 4,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Recupera memoria por scope, vigencia, tipo y relevancia léxica.

        El aislamiento se aplica antes del ranking. Luego se filtran recuerdos
        expirados y, si hay query, se usa overlap léxico + confianza + prioridad.
        Los empates favorecen la versión actualizada más recientemente.
        """
        if limit < 0:
            raise ValueError("limit debe ser >= 0")
        now = now or utc_now()

        query_tokens = _tokens(query or "")

        candidates = (self.list_scope(scope))

        ranked = []

        for item in candidates:
            expires_at = (_parse_iso(item["expires_at"]))

            if (
                expires_at is not None
                and expires_at <= now
            ):
                continue

            if (
                kinds is not None
                and item["kind"]
                not in kinds
            ):
                continue

            haystack = (
                item["key"]
                + " "
                + json.dumps(
                    item["value"],
                    ensure_ascii=False,
                )
            )

            hay_tokens = _tokens(haystack)

            overlap = (
                (
                    len(
                        query_tokens
                        & hay_tokens
                    )
                    / len(query_tokens)
                )
                if query_tokens
                else 1.0
            )

            if (
                query_tokens
                and overlap == 0
            ):
                continue

            score = (
                overlap * 10
                + float(item["confidence"])
                + int(item["priority"]) * 0.1
            )

            ranked.append({
                **item,
                "retrieve_score":
                    round(
                        score,
                        4,
                    ),
            })

        ranked.sort(
            key=lambda item: (
                -item["retrieve_score"],
                -(
                    _parse_iso(item["updated_at"])
                    or datetime.min.replace(tzinfo=timezone.utc)
                ).timestamp(),
            )
        )

        return ranked[:limit]

    def supersede(
        self,
        old_memory_id: str,
        *,
        scope: MemoryScope,
        candidate: MemoryCandidate,
        now: datetime | None = None,
    ) -> str:
        """Reemplaza un recuerdo activo preservando trazabilidad histórica.

        La validación del candidato, el cambio de estado del recuerdo anterior
        y la inserción de la nueva versión ocurren en una única transacción.
        Si algo falla, SQLite revierte todo y no deja una preferencia "perdida".
        """
        decision = evaluate_write(candidate)

        if not decision.accepted:
            raise ValueError(
                "Memory candidate rechazado: "
                + decision.reason
            )

        now = now or utc_now()

        with self._connection() as conn:
            old = conn.execute(
                """
                SELECT memory_id
                FROM memory
                WHERE memory_id=?
                  AND user_id=?
                  AND project_id=?
                  AND course_id=?
                  AND session_id=?
                  AND status='active'
                """,
                (
                    old_memory_id,
                    scope.user_id,
                    scope.project_id,
                    scope.course_id,
                    scope.session_id,
                ),
            ).fetchone()

            if old is None:
                raise KeyError(
                    "Recuerdo activo no "
                    "encontrado en ese scope"
                )

            conn.execute(
                """
                UPDATE memory
                SET status='superseded',
                    updated_at=?
                WHERE memory_id=?
                """,
                (
                    _iso(now),
                    old_memory_id,
                ),
            )

            new_memory_id = (
                self._insert_active(
                    conn,
                    scope=scope,
                    candidate=candidate,
                    supersedes=(old_memory_id),
                    now=now,
                )
            )

        return new_memory_id

    def delete(
        self,
        memory_id: str,
        *,
        scope: MemoryScope,
    ):
        """Marca un recuerdo como borrado dentro del scope indicado.

        Usamos soft delete para conservar trazabilidad. Los métodos normales de
        recuperación no devuelven registros con ``status='deleted'``.
        """
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE memory
                SET status='deleted',
                    updated_at=?
                WHERE memory_id=?
                  AND user_id=?
                  AND project_id=?
                  AND course_id=?
                  AND session_id=?
                """,
                (
                    _iso(utc_now()),
                    memory_id,
                    scope.user_id,
                    scope.project_id,
                    scope.course_id,
                    scope.session_id,
                ),
            )

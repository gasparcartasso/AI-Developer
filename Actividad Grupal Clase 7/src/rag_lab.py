"""Helpers para la actividad grupal de Clase 7.

La actividad NO ejecuta embeddings ni Qdrant. Los resultados de retrieval
están congelados para que toda diferencia provenga de decisiones de C7.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


ACTIVITY_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ACTIVITY_DIR / "data"

INJECTION_PATTERNS = [
    r"ignor[aá] (?:todas? )?(?:las )?instrucciones",
    r"system prompt",
    r"revel[aá] (?:el )?prompt",
    r"actu[aá] como",
]


@dataclass
class Candidate:
    rank: int
    chunk_id: str
    document_id: str
    source: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextResult:
    context: str
    used: list[Candidate]
    dropped: list[dict[str, str]]

    @property
    def context_chars(self) -> int:
        return len(self.context)


@dataclass
class ValidationResult:
    valid: bool
    citations: list[tuple[str, str]]
    errors: list[str]


@dataclass
class CaseResult:
    case_id: str
    case_type: str
    query: str
    expected_mode: str
    predicted_mode: str
    retrieved: list[Candidate]
    context: ContextResult
    answer: str
    validation: ValidationResult

    def as_dict(self) -> dict[str, Any]:
        required = set()
        forbidden = set()
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "query": self.query,
            "expected_mode": self.expected_mode,
            "predicted_mode": self.predicted_mode,
            "retrieved_count": len(self.retrieved),
            "used_chunk_ids": [item.chunk_id for item in self.context.used],
            "dropped": self.context.dropped,
            "context_chars": self.context.context_chars,
            "answer": self.answer,
            "citations": [
                {"source": source, "chunk_id": chunk_id}
                for source, chunk_id in self.validation.citations
            ],
            "validation_ok": self.validation.valid,
            "validation_errors": self.validation.errors,
        }


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or DATA_DIR / "frozen_retrieval_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_config(path: Path | None = None) -> dict[str, Any]:
    path = path or DATA_DIR / "baseline_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def to_candidates(raw: list[dict[str, Any]]) -> list[Candidate]:
    return [Candidate(**item) for item in raw]


def looks_like_instruction_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


CITATION_RE = re.compile(
    r"\[SOURCE:\s*(?P<source>[^|\]]+?)\s*\|\s*CHUNK:\s*(?P<chunk>[^\]]+?)\s*\]"
)


def parse_citations(answer: str) -> list[tuple[str, str]]:
    return [
        (match.group("source").strip(), match.group("chunk").strip())
        for match in CITATION_RE.finditer(answer)
    ]


def format_block(candidate: Candidate) -> str:
    return (
        f"[SOURCE: {candidate.source} | CHUNK: {candidate.chunk_id}]\n"
        f"{candidate.text.strip()}"
    )


# ---------------------------------------------------------------------------
# Baseline intencionalmente imperfecto
# ---------------------------------------------------------------------------

def baseline_build_context(
    candidates: list[Candidate],
    config: dict[str, Any],
) -> ContextResult:
    parts: list[str] = []
    used: list[Candidate] = []
    dropped: list[dict[str, str]] = []
    total = 0

    for candidate in candidates:
        if candidate.score < config["inherited_min_score"]:
            dropped.append({"chunk_id": candidate.chunk_id, "reason": "below_inherited_threshold"})
            continue

        block = format_block(candidate)

        if total + len(block) > config["max_context_chars"]:
            dropped.append({"chunk_id": candidate.chunk_id, "reason": "budget_exceeded"})
            # BUG deliberado para diagnosticar en la actividad.
            break

        parts.append(block)
        used.append(candidate)
        total += len(block) + (2 if parts else 0)

    return ContextResult(
        context="\n\n".join(parts),
        used=used,
        dropped=dropped,
    )


def baseline_decide_mode(
    query: str,
    context: ContextResult,
    config: dict[str, Any],
) -> str:
    # Baseline deliberadamente simple: no distingue ambigüedad.
    return "ANSWER" if context.used else "ABSTAIN"


def frozen_generator(
    case: dict[str, Any],
    context: ContextResult,
    retrieved: list[Candidate],
) -> str:
    mode = case.get("generator_behavior", "normal")

    if not context.used:
        return "No tengo evidencia suficiente."

    if mode == "malformed_citation":
        item = context.used[0]
        return f"{item.text} [SOURCE {item.source} CHUNK {item.chunk_id}]"

    if mode == "invalid_citation":
        item = context.used[0]
        return (
            f"{item.text} "
            "[SOURCE: ghost.md | CHUNK: GHOST:0]"
        )

    if mode == "cite_retrieved_not_used":
        cited = retrieved[-1]
        item = context.used[0]
        return (
            f"{item.text} "
            f"[SOURCE: {cited.source} | CHUNK: {cited.chunk_id}]"
        )

    selected = context.used[:3]
    return " ".join(
        f"{item.text} [SOURCE: {item.source} | CHUNK: {item.chunk_id}]"
        for item in selected
    )


def baseline_validate(
    answer: str,
    context: ContextResult,
    retrieved: list[Candidate],
    config: dict[str, Any],
) -> ValidationResult:
    citations = parse_citations(answer)
    errors: list[str] = []

    if "[SOURCE" in answer and not citations:
        errors.append("Cita malformada")

    # BUG deliberado: valida contra retrieved, no contra context set.
    allowed = {(item.source, item.chunk_id) for item in retrieved}

    for citation in citations:
        if citation not in allowed:
            errors.append(
                f"Cita fuera del conjunto permitido: {citation[0]} / {citation[1]}"
            )

    if not citations and answer != "No tengo evidencia suficiente.":
        errors.append("Respuesta sin citas")

    return ValidationResult(
        valid=not errors,
        citations=citations,
        errors=errors,
    )


def run_baseline_case(
    case: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> CaseResult:
    config = config or load_baseline_config()
    retrieved = to_candidates(case["candidates"])
    context = baseline_build_context(retrieved, config)
    mode = baseline_decide_mode(case["query"], context, config)

    if mode != "ANSWER":
        answer = (
            "Necesito más información."
            if mode == "CLARIFY"
            else "No tengo evidencia suficiente."
        )
    else:
        answer = frozen_generator(case, context, retrieved)

    validation = baseline_validate(
        answer,
        context,
        retrieved,
        config,
    )

    return CaseResult(
        case_id=case["case_id"],
        case_type=case["case_type"],
        query=case["query"],
        expected_mode=case["expected_mode"],
        predicted_mode=mode,
        retrieved=retrieved,
        context=context,
        answer=answer,
        validation=validation,
    )


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate_results(
    cases: list[dict[str, Any]],
    results: list[CaseResult],
) -> dict[str, Any]:
    by_id = {case["case_id"]: case for case in cases}
    rows = []

    total_required = 0
    total_required_used = 0
    forbidden_total = 0
    validator_correct_cases = 0

    for result in results:
        case = by_id[result.case_id]
        required = set(case.get("required_chunk_ids", []))
        forbidden = set(case.get("forbidden_context_chunk_ids", []))
        used = {item.chunk_id for item in result.context.used}

        required_used = len(required & used)
        forbidden_used = len(forbidden & used)

        total_required += len(required)
        total_required_used += required_used
        forbidden_total += forbidden_used
        expected_validation = bool(case.get("expected_validation", True))
        validator_correct_cases += int(
            result.validation.valid == expected_validation
        )

        rows.append({
            "case_id": result.case_id,
            "case_type": result.case_type,
            "expected_mode": result.expected_mode,
            "predicted_mode": result.predicted_mode,
            "mode_ok": result.predicted_mode == result.expected_mode,
            "required_context_recall": (
                required_used / len(required)
                if required
                else 1.0
            ),
            "forbidden_in_context": forbidden_used,
            "context_chars": result.context.context_chars,
            "validation_ok": result.validation.valid,
            "expected_validation": bool(case.get("expected_validation", True)),
            "validator_ok": (
                result.validation.valid
                == bool(case.get("expected_validation", True))
            ),
            "used_chunk_ids": [item.chunk_id for item in result.context.used],
            "errors": result.validation.errors,
        })

    n = max(len(results), 1)

    summary = {
        "n_cases": len(results),
        "mode_accuracy": sum(row["mode_ok"] for row in rows) / n,
        "required_evidence_recall": (
            total_required_used / total_required
            if total_required
            else 1.0
        ),
        "unsafe_or_forbidden_context_count": forbidden_total,
        "validator_accuracy": validator_correct_cases / n,
        "avg_context_chars": (
            sum(row["context_chars"] for row in rows) / n
        ),
    }

    return {
        "summary": summary,
        "cases": rows,
    }


def evaluate_baseline(
    cases: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cases = cases or load_cases()
    config = config or load_baseline_config()
    results = [run_baseline_case(case, config) for case in cases]
    return evaluate_results(cases, results)


def evaluate_team(
    run_case: Callable[[dict[str, Any]], CaseResult],
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cases = cases or load_cases()
    results = [run_case(case) for case in cases]
    return evaluate_results(cases, results)


def inspect_case(
    case_id: str,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cases = cases or load_cases()
    case = next(item for item in cases if item["case_id"] == case_id)
    return {
        **{k: v for k, v in case.items() if k != "candidates"},
        "candidates": case["candidates"],
    }

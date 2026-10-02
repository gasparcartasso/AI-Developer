import re
from typing import Any

from pydantic import ValidationError

from .schemas import TicketClassification
from .prompting import build_zero_shot_prompt

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def validate_user_text(text: str, *, min_len: int = 12, max_len: int = 2_000) -> str:
    if not isinstance(text, str):
        raise TypeError("El texto debe ser str")
    cleaned = text.strip()
    if len(cleaned) < min_len:
        raise ValueError("Input demasiado corto para clasificar con confianza")
    if len(cleaned) > max_len:
        raise ValueError("Input demasiado largo para este clasificador")
    if CONTROL_CHARS_RE.search(cleaned):
        raise ValueError("Input contiene caracteres de control no permitidos")
    return cleaned


def fallback_classification(reason: str) -> TicketClassification:
    return TicketClassification(
        intent="general",
        priority="media",
        needs_human=True,
        confidence=0.0,
        summary=f"Fallback seguro: {reason[:120]}",
        evidence=["fallback sin evidencia confiable"],
    )


def apply_business_rules(result: TicketClassification, original_text: str) -> TicketClassification:
    text = original_text.lower()
    updates: dict[str, Any] = {}
    sensitive_terms = ("api key", "password", "contraseña", "credencial", "phishing")
    if any(term in text for term in sensitive_terms):
        updates["intent"] = "seguridad"
        updates["priority"] = "alta"
        updates["needs_human"] = True
    if result.confidence < 0.70:
        updates["needs_human"] = True
    if result.priority == "alta":
        updates["needs_human"] = True
    return result.model_copy(update=updates) if updates else result


def classify_with_retry(text: str, model_call, *, prompt_builder=build_zero_shot_prompt, max_attempts: int = 2) -> TicketClassification:
    try:
        clean_text = validate_user_text(text)
    except Exception as exc:
        return fallback_classification(str(exc))

    prompt = prompt_builder(clean_text)
    for _ in range(max_attempts):
        raw = model_call(prompt)
        try:
            parsed = TicketClassification.model_validate_json(raw)
            return apply_business_rules(parsed, clean_text)
        except ValidationError:
            prompt += "\nLa salida anterior no respetó el schema. Devolvé SOLO JSON válido."
    return fallback_classification("salida inválida luego de retry")

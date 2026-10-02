from __future__ import annotations

from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class RoutingDecision:
    provider: str
    reason: str
    requires_human_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

SENSITIVE_TERMS = {
    "dni", "documento", "salario", "sueldo", "contraseña",
    "password", "historia clínica", "tarjeta", "cuenta bancaria",
}

def contains_sensitive_data(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in SENSITIVE_TERMS)

def choose_provider_for_task(
    *,
    text: str,
    needs_tools: bool,
    complexity: str = "low",
    local_available: bool = True,
    cloud_available: bool = True,
) -> RoutingDecision:
    if complexity not in {"low", "medium", "high"}:
        raise ValueError("complexity debe ser low, medium o high")

    sensitive = contains_sensitive_data(text)

    if sensitive:
        if local_available:
            requires_review = needs_tools or complexity == "high"
            reason = (
                "datos sensibles; validar capacidad local y mantener revisi?n humana"
                if requires_review
                else "datos sensibles y capacidad local suficiente"
            )
            return RoutingDecision(
                "local", reason, requires_human_review=requires_review
            )
        if cloud_available:
            return RoutingDecision(
                "cloud",
                "local no disponible; autorizaci?n expl?cita requerida antes de enviar datos",
                requires_human_review=True,
            )
        raise RuntimeError("No hay providers disponibles")

    if needs_tools or complexity == "high":
        if cloud_available:
            return RoutingDecision("cloud", "requiere tools o razonamiento complejo")
        if local_available:
            return RoutingDecision(
                "local",
                "cloud no disponible; fallback con revisi?n humana",
                requires_human_review=True,
            )
        raise RuntimeError("No hay providers disponibles")

    if local_available:
        return RoutingDecision("local", "tarea simple; prioriza costo y disponibilidad local")
    if cloud_available:
        return RoutingDecision("cloud", "local no disponible")
    raise RuntimeError("No hay providers disponibles")

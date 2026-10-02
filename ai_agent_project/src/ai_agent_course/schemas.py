from typing import Literal
from pydantic import BaseModel, Field


class TicketClassification(BaseModel):
    intent: Literal["vacaciones", "gastos", "seguridad", "trabajo_remoto", "general"] = Field(
        description="Categoría principal de la solicitud."
    )
    priority: Literal["baja", "media", "alta"] = Field(
        description="Urgencia operativa de la solicitud."
    )
    needs_human: bool = Field(
        description="True si hay ambigüedad, riesgo o necesidad de intervención humana."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=8, max_length=180)
    evidence: list[str] = Field(min_length=1, max_length=3)

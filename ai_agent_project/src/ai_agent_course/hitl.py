from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import uuid

from pydantic import BaseModel, Field

RunStatus = Literal[
    "new",
    "waiting_for_approval",
    "approved_pending_policy",
    "executing",
    "needs_recovery",
    "completed",
    "rejected",
    "failed",
]

DecisionKind = Literal["approve", "edit", "reject"]
RiskLevel = Literal["low", "medium", "high"]


def utc_now_iso() -> str:
    """Devuelve el instante actual en UTC como texto ISO-8601.

    Returns:
        Timestamp con zona horaria. Se usa en el event log para conservar un
        orden temporal auditable sin depender de la zona horaria local.
    """
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    """Calcula una huella SHA-256 estable para un valor JSON-serializable.

    Args:
        value: Datos que queremos identificar de forma estable.

    Returns:
        Hash hexadecimal completo. En esta clase liga una aprobación humana a
        los argumentos exactos de la propuesta que la persona revisó.
    """
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ProposedAction(BaseModel):
    action_id: str = Field(default_factory=lambda: "act-" + str(uuid.uuid4()))
    tool_name: str
    args: dict[str, Any]
    rationale: str
    evidence_used: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "medium"
    version: int = Field(default=1, ge=1)
    args_hash: str
    idempotency_key: str


class HumanDecision(BaseModel):
    decision: DecisionKind
    actor: str = Field(min_length=2)
    action_id: str
    proposal_version: int = Field(ge=1)
    args_hash: str
    edited_args: dict[str, Any] | None = None
    comment: str = ""


class AgentRunState(BaseModel):
    run_id: str = Field(default_factory=lambda: "run-" + str(uuid.uuid4()))
    thread_id: str
    user_message: str
    context_capsule: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus = "new"
    pending_action: ProposedAction | None = None
    last_decision: HumanDecision | None = None
    result: dict[str, Any] | None = None
    checkpoint_version: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)


class StaleCheckpointError(RuntimeError):
    """El State parte de una versión anterior a la última persistida."""


class DecisionMismatchError(ValueError):
    """La decisión humana no apunta a la propuesta actualmente pendiente."""


class InvalidTransitionError(ValueError):
    """El workflow intentó una transición no permitida por su State actual."""


class JsonCheckpointStore:
    """Store JSON didáctico de snapshots versionados por ``run_id``.

    El control de versión permite detectar snapshots obsoletos en el flujo
    secuencial de esta notebook. No sustituye un compare-and-swap transaccional
    entre procesos concurrentes.
    """

    def __init__(self, path: str | Path):
        """Inicializa el archivo de checkpoints.

        Args:
            path: Ruta del JSON durable usado por el laboratorio.

        Returns:
            ``None``. Crea el archivo si todavía no existe.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def save(self, state: AgentRunState) -> AgentRunState:
        """Persiste un State y avanza su ``checkpoint_version``.

        Args:
            state: Snapshot que declara la versión desde la que fue leído.

        Returns:
            Copia del State con la nueva versión persistida.

        Raises:
            StaleCheckpointError: Si el store ya avanzó a otra versión.
        """
        data = json.loads(self.path.read_text(encoding="utf-8"))
        current = data.get(state.run_id)
        current_version = current["version"] if current else 0

        if state.checkpoint_version != current_version:
            raise StaleCheckpointError(
                f"snapshot viejo: state={state.checkpoint_version}, "
                f"store={current_version}"
            )

        new_version = current_version + 1
        saved = state.model_copy(update={"checkpoint_version": new_version})
        data[state.run_id] = {
            "version": new_version,
            "state": saved.model_dump(),
        }
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return saved

    def load(self, run_id: str) -> AgentRunState:
        """Recupera y valida el último State persistido de un run.

        Args:
            run_id: Identidad lógica del run.

        Returns:
            ``AgentRunState`` listo para continuar el workflow.

        Raises:
            KeyError: Si el ``run_id`` nunca fue persistido.
        """
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if run_id not in data:
            raise KeyError(f"Run inexistente: {run_id}")
        return AgentRunState.model_validate(data[run_id]["state"])


def add_event(state: AgentRunState, event: str, **metadata: Any) -> None:
    """Agrega un hecho auditable al event log del State.

    Args:
        state: Run cuyo historial queremos ampliar.
        event: Nombre estable del hecho ocurrido.
        **metadata: Datos específicos del evento.

    Returns:
        ``None``. Modifica ``state.events`` in-place para que el siguiente
        checkpoint conserve la traza.
    """
    state.events.append({
        "event_id": "evt-" + str(uuid.uuid4()),
        "event": event,
        "timestamp": utc_now_iso(),
        **metadata,
    })


def build_action(
    *,
    tool_name: str,
    args: dict[str, Any],
    rationale: str,
    evidence_used: list[str],
    risk_level: RiskLevel,
    action_id: str | None = None,
    version: int = 1,
) -> ProposedAction:
    """Construye una propuesta versionada antes de ejecutar un side effect.

    Args:
        tool_name: Tool que eventualmente realizará el efecto.
        args: Argumentos concretos que el humano verá y aprobará.
        rationale: Motivo por el que el agente propone la acción.
        evidence_used: Evidencia que fundamenta la propuesta.
        risk_level: Nivel de riesgo visible para la revisión.
        action_id: Identidad lógica estable de la acción; se crea si falta.
        version: Versión de la propuesta.

    Returns:
        ``ProposedAction`` con hash de argumentos e identidad de idempotencia.
    """
    action_id = action_id or "act-" + str(uuid.uuid4())
    return ProposedAction(
        action_id=action_id,
        tool_name=tool_name,
        args=dict(args),
        rationale=rationale,
        evidence_used=list(evidence_used),
        risk_level=risk_level,
        version=version,
        args_hash=canonical_hash(args),
        # La key identifica el EFECTO EXACTO y no solo un action_id genérico.
        # Un Edit crea v2 con nuevos argumentos y, por tanto, nueva key.
        idempotency_key=(
            f"{tool_name}:{action_id}:v{version}:{canonical_hash(args)}"
        ),
    )


def prepare_action(
    state: AgentRunState,
    *,
    tool_name: str,
    args: dict[str, Any],
    rationale: str,
    evidence_used: list[str],
    risk_level: RiskLevel = "medium",
) -> AgentRunState:
    """Prepara una acción y mueve el run al Human Gate.

    Args:
        state: Run en estado ``new``.
        tool_name: Tool propuesta.
        args: Argumentos aún no ejecutados.
        rationale: Justificación mostrada al revisor.
        evidence_used: Evidencia usada para proponer la acción.
        risk_level: Riesgo visible para el revisor.

    Returns:
        El mismo State actualizado a ``waiting_for_approval``.

    Raises:
        InvalidTransitionError: Si el run ya había avanzado desde ``new``.
    """
    if state.status != "new":
        raise InvalidTransitionError(f"prepare no permitido desde {state.status}")

    state.pending_action = build_action(
        tool_name=tool_name,
        args=args,
        rationale=rationale,
        evidence_used=evidence_used,
        risk_level=risk_level,
    )
    state.status = "waiting_for_approval"
    add_event(
        state,
        "approval_requested",
        action_id=state.pending_action.action_id,
        proposal_version=state.pending_action.version,
        args_hash=state.pending_action.args_hash,
        risk_level=state.pending_action.risk_level,
    )
    return state


def _assert_decision_targets_current_proposal(
    action: ProposedAction,
    decision: HumanDecision,
) -> None:
    """Verifica que la decisión corresponda exactamente a la propuesta vigente.

    Args:
        action: Propuesta actualmente pendiente.
        decision: Decisión humana recibida.

    Returns:
        ``None`` si ``action_id``, versión y hash coinciden.

    Raises:
        DecisionMismatchError: Si la decisión pertenece a otra propuesta o versión.
    """
    if decision.action_id != action.action_id:
        raise DecisionMismatchError("action_id no coincide con la propuesta actual")
    if decision.proposal_version != action.version:
        raise DecisionMismatchError("proposal_version obsoleta")
    if decision.args_hash != action.args_hash:
        raise DecisionMismatchError("args_hash no coincide con la propuesta actual")


def apply_human_decision(
    state: AgentRunState,
    decision_payload: dict[str, Any],
    *,
    editable_fields: set[str] | None = None,
) -> AgentRunState:
    """Aplica ``approve``, ``edit`` o ``reject`` sobre la propuesta vigente.

    Args:
        state: Run detenido en ``waiting_for_approval``.
        decision_payload: Decisión estructurada enviada por el revisor.
        editable_fields: Campos que un ``edit`` puede modificar. ``None`` usa
            todos los argumentos actuales; un set vacío significa "ninguno".

    Returns:
        State actualizado. ``edit`` crea una nueva versión y vuelve a esperar;
        ``approve`` avanza a Policy; ``reject`` finaliza sin ejecutar.

    Raises:
        InvalidTransitionError: Si no estamos esperando aprobación.
        DecisionMismatchError: Si la decisión apunta a una propuesta vieja.
        ValueError: Si ``edit`` no trae cambios o toca campos no editables.
    """
    if state.status != "waiting_for_approval":
        raise InvalidTransitionError(
            f"el run no espera aprobación: {state.status}"
        )
    if state.pending_action is None:
        raise InvalidTransitionError("no existe pending_action")

    action = state.pending_action
    decision = HumanDecision.model_validate(decision_payload)
    _assert_decision_targets_current_proposal(action, decision)

    # Comprobar primero toda la edición evita un State parcialmente mutado
    # si el payload es inválido (el evento tampoco debe quedar registrado).
    if decision.decision == "edit":
        if not decision.edited_args:
            raise ValueError("edit requiere edited_args")
        allowed = set(action.args) if editable_fields is None else set(editable_fields)
        forbidden = set(decision.edited_args) - allowed
        if forbidden:
            raise ValueError("campos no editables: " + ", ".join(sorted(forbidden)))

    state.last_decision = decision
    add_event(
        state,
        "human_decision",
        decision=decision.decision,
        actor=decision.actor,
        action_id=action.action_id,
        proposal_version=action.version,
        args_hash=action.args_hash,
        comment=decision.comment,
    )

    if decision.decision == "reject":
        state.status = "rejected"
        return state

    if decision.decision == "edit":
        # Los argumentos ya fueron validados antes de tocar el State.
        new_args = {**action.args, **decision.edited_args}
        new_action = build_action(
            tool_name=action.tool_name,
            args=new_args,
            rationale=action.rationale,
            evidence_used=action.evidence_used,
            risk_level=action.risk_level,
            action_id=action.action_id,
            version=action.version + 1,
            # No reutilizar la key de una propuesta con argumentos anteriores.
        )
        state.pending_action = new_action
        state.status = "waiting_for_approval"
        add_event(
            state,
            "proposal_edited",
            action_id=new_action.action_id,
            old_version=action.version,
            new_version=new_action.version,
            args_hash=new_action.args_hash,
        )
        return state

    state.status = "approved_pending_policy"
    return state


def mark_policy_authorized(
    state: AgentRunState,
    *,
    validated_args: dict[str, Any],
) -> AgentRunState:
    """Registra que Validation + Policy habilitaron la propuesta aprobada.

    Args:
        state: Run en ``approved_pending_policy``.
        validated_args: Argumentos ya validados por el schema de la Tool.

    Returns:
        State en ``executing`` con un evento de autorización.
    """
    if state.status != "approved_pending_policy":
        raise InvalidTransitionError(
            f"Policy no esperada desde {state.status}"
        )
    state.status = "executing"
    add_event(
        state,
        "policy_authorized",
        action_id=state.pending_action.action_id if state.pending_action else None,
        validated_args_hash=canonical_hash(validated_args),
    )
    return state


def mark_policy_denied(
    state: AgentRunState,
    *,
    reason: str,
) -> AgentRunState:
    """Finaliza el run cuando Policy rechaza una propuesta ya aprobada.

    Args:
        state: Run en ``approved_pending_policy``.
        reason: Motivo devuelto por la Policy.

    Returns:
        State en ``failed`` con evidencia de la denegación.
    """
    if state.status != "approved_pending_policy":
        raise InvalidTransitionError(
            f"Policy no esperada desde {state.status}"
        )
    state.status = "failed"
    add_event(state, "policy_denied", reason=reason)
    return state

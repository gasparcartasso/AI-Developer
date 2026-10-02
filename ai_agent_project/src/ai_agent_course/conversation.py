from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    created_at: str

    @classmethod
    def create(cls, role: str, content: str) -> "Message":
        if role not in {"user", "assistant"}:
            raise ValueError("role debe ser user o assistant")
        if not content.strip():
            raise ValueError("content no puede estar vacío")
        return cls(
            role=role,
            content=content.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConversationSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_turns: int = 6
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> Message:
        message = Message.create(role, content)
        self.messages.append(message)
        return message

    def recent_messages(self) -> list[Message]:
        return self.messages[-self.max_turns * 2:]

    def build_prompt(self, user_text: str) -> str:
        if not user_text.strip():
            raise ValueError("user_text no puede estar vacío")
        history = "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in self.recent_messages()
        ) or "(sin historial previo)"
        return f"""
HISTORIAL RECIENTE
{history}

NUEVO MENSAJE DEL USUARIO
{user_text.strip()}

Respondé considerando el historial reciente.
""".strip()

    def transcript(self) -> list[dict]:
        return [message.to_dict() for message in self.messages]

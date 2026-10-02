from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    run_id: str
    session_id: str
    event: str
    timestamp: str
    payload: dict


class JsonlTraceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, run_id: str, session_id: str, event: str, payload: dict) -> TraceEvent:
        item = TraceEvent(
            event_id=str(uuid.uuid4()),
            run_id=run_id,
            session_id=session_id,
            event=event,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload,
        )
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(item), ensure_ascii=False, default=str) + "\n")
        return item

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

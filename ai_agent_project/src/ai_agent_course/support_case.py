from __future__ import annotations

import json
from pathlib import Path


def load_support_case(project_dir):
    root = Path(project_dir) / "data" / "mock_gmail"

    threads_path = root / "threads.json"
    kb_path = root / "knowledge_base.md"

    return (
        json.loads(
            threads_path.read_text(
                encoding="utf-8"
            )
        ),
        kb_path.read_text(
            encoding="utf-8"
        ),
    )

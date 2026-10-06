from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml


_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _normalize(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _split_metadata(value: Any) -> list[str]:
    return [
        item.strip()
        for item in str(value or "").split("|")
        if item.strip()
    ]


@dataclass(frozen=True)
class SkillSelection:
    selected: str | None
    reason: str
    candidates: list[dict[str, Any]]


class SkillLoader:
    """Loader didáctico compatible con el formato Agent Skills."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.events: list[dict[str, Any]] = []

    def reset_trace(self):
        self.events.clear()

    def _skill_path(self, name: str) -> Path:
        directory = (self.root / name).resolve()

        if directory.parent != self.root.resolve():
            raise ValueError("La Skill debe ser una carpeta directa de la raíz")

        return directory / "SKILL.md"

    def _validate_frontmatter(
        self,
        path: Path,
        frontmatter: dict,
    ) -> None:
        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not isinstance(name, str) or not _SKILL_NAME.fullmatch(name):
            raise ValueError(f"Nombre Agent Skill inválido: {name!r}")

        if name != path.parent.name:
            raise ValueError(
                "El name del frontmatter debe coincidir con la carpeta"
            )

        if not isinstance(description, str) or not (1 <= len(description) <= 1024):
            raise ValueError("description debe tener entre 1 y 1024 caracteres")

        metadata = frontmatter.get("metadata", {})
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata debe ser un mapping")

        for key, value in (metadata or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    "Agent Skills define metadata como string → string"
                )

    def _read_frontmatter(self, path: Path) -> dict:
        """Lee solo metadata: no consume el cuerpo Markdown."""
        lines = []

        with path.open("r", encoding="utf-8") as handle:
            if handle.readline().strip() != "---":
                raise ValueError(f"{path} no comienza con frontmatter")

            for line in handle:
                if line.strip() == "---":
                    break
                lines.append(line)
            else:
                raise ValueError(f"{path} no cierra el frontmatter")

        frontmatter = yaml.safe_load("".join(lines)) or {}
        self._validate_frontmatter(path, frontmatter)

        course = frontmatter.get("metadata") or {}

        self.events.append({
            "event": "metadata_read",
            "skill": frontmatter["name"],
            "version": str(course.get("course-version", "unversioned")),
        })

        return frontmatter

    def _entry_from_frontmatter(self, frontmatter: dict) -> dict:
        course = frontmatter.get("metadata") or {}

        return {
            "name": frontmatter["name"],
            "description": frontmatter["description"],
            "version": str(course.get("course-version", "unversioned")),
            "activation": {
                "when": _split_metadata(course.get("course-activation")),
                "when_not": _split_metadata(
                    course.get("course-negative-signals")
                ),
            },
            "tool_hints": _split_metadata(
                course.get("course-tool-hints")
            ),
            "resources": _split_metadata(
                course.get("course-resources")
            ),
            "tags": _split_metadata(course.get("course-tags")),
        }

    def list_summaries(self) -> list[dict]:
        summaries = []

        for path in sorted(self.root.glob("*/SKILL.md")):
            summaries.append(
                self._entry_from_frontmatter(
                    self._read_frontmatter(path)
                )
            )

        return summaries

    def _score(
        self,
        query: str,
        entry: dict,
    ) -> tuple[int, list[str], list[str]]:
        """Selector léxico explicable usado solo con fines didácticos."""
        text = _normalize(query)
        activation = entry.get("activation", {})

        positive = [
            _normalize(item)
            for item in activation.get("when", [])
        ]
        negative = [
            _normalize(item)
            for item in activation.get("when_not", [])
        ]

        matched_positive = [
            item for item in positive
            if item and item in text
        ]
        matched_negative = [
            item for item in negative
            if item and item in text
        ]

        score = (
            2 * len(matched_positive)
            - 3 * len(matched_negative)
        )

        for tag in entry.get("tags", []):
            if _normalize(tag) in text:
                score += 1

        return score, matched_positive, matched_negative

    def shortlist(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict]:
        if limit < 0:
            raise ValueError("limit debe ser >= 0")

        rows = []

        for entry in self.list_summaries():
            score, pos, neg = self._score(query, entry)

            if score > 0:
                rows.append({
                    "name": entry["name"],
                    "version": entry["version"],
                    "description": entry["description"],
                    "score": score,
                    "matched_positive": pos,
                    "matched_negative": neg,
                })

        rows.sort(
            key=lambda item: (-item["score"], item["name"])
        )

        return rows[:limit]

    def choose(
        self,
        query: str,
        *,
        min_score: int = 2,
        ambiguity_margin: int = 0,
    ) -> SkillSelection:
        candidates = self.shortlist(query)

        if not candidates:
            return SkillSelection(
                selected=None,
                reason="fallback:no_candidate",
                candidates=[],
            )

        top = candidates[0]

        if top["score"] < min_score:
            return SkillSelection(
                selected=None,
                reason="fallback:low_confidence",
                candidates=candidates,
            )

        if len(candidates) > 1:
            second = candidates[1]

            if top["score"] - second["score"] <= ambiguity_margin:
                return SkillSelection(
                    selected=None,
                    reason="fallback:ambiguous",
                    candidates=candidates,
                )

        return SkillSelection(
            selected=top["name"],
            reason="selected",
            candidates=candidates,
        )

    def load(self, name: str) -> dict:
        """Carga Nivel 2: frontmatter + cuerpo completo de SKILL.md."""
        path = self._skill_path(name)
        frontmatter = self._read_frontmatter(path)
        entry = self._entry_from_frontmatter(frontmatter)

        text = path.read_text(encoding="utf-8")
        _, _, body = text.split("---", 2)
        body = body.strip()

        if not body:
            raise ValueError("SKILL.md debe contener instrucciones")

        self.events.append({
            "event": "skill_body_load",
            "skill": name,
            "version": entry["version"],
        })

        return {
            **entry,
            "frontmatter": frontmatter,
            "instructions": body,
        }

    def load_resource(
        self,
        name: str,
        relative_path: str,
    ) -> str:
        """Carga Nivel 3 sin permitir escapar del directorio de la Skill."""
        directory = self._skill_path(name).parent
        resource = (directory / relative_path).resolve()

        if resource != directory and directory not in resource.parents:
            raise ValueError("Ruta de recurso fuera de la Skill")

        if not resource.is_file():
            raise FileNotFoundError(relative_path)

        text = resource.read_text(encoding="utf-8")

        self.events.append({
            "event": "resource_load",
            "skill": name,
            "resource": relative_path,
        })

        return text

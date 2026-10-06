from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estima tokens con una regla simple de ~4 caracteres por token.

    Parameters
    ----------
    text:
        Texto cuyo costo aproximado queremos estimar.

    Returns
    -------
    int
        Cantidad estimada de tokens.

    Es una aproximación didáctica. En producción conviene usar el tokenizer del
    modelo/proveedor porque distintos modelos tokenizan de manera diferente.
    """
    text = str(text)

    if not text:
        return 0

    return max(
        1,
        math.ceil(len(text) / 4),
    )


def truncate_to_tokens(
    text: str,
    max_tokens: int,
) -> str:
    """Trunca texto usando la misma estimación didáctica de tokens.

    Devuelve el texto original si ya entra en ``max_tokens``. Si no entra,
    recorta por caracteres y agrega ``…`` para hacer visible la pérdida.
    """
    if max_tokens <= 0:
        return ""

    if (
        estimate_tokens(text)
        <= max_tokens
    ):
        return text

    max_chars = max(
        0,
        max_tokens * 4,
    )

    if max_chars <= 1:
        return ""

    return (
        text[: max_chars - 1].rstrip()
        + "…"
    )


def deduplicate_text_blocks(blocks: list[str],) -> list[str]:
    """Elimina bloques textuales repetidos preservando el primer orden.

    La comparación ignora diferencias de mayúsculas y espacios. Esto reduce
    context rot y gasto de tokens sin alterar la primera evidencia observada.
    """
    seen = set()
    result = []

    for block in blocks:
        normalized = " ".join(
            str(block)
            .casefold()
            .split()
        )

        if (
            not normalized
            or normalized in seen
        ):
            continue

        seen.add(normalized)
        result.append(str(block))

    return result


@dataclass
class ContextItem:
    """Unidad candidata a entrar en el contexto del modelo.

    ``priority`` ordena importancia, ``required`` prohíbe descartarla y
    ``max_tokens`` limita cuánto contenido de esa sección puede ocupar.
    """

    name: str
    text: str
    priority: int
    required: bool = False
    allow_truncate: bool = True
    max_tokens: int | None = None


class ContextBuilder:
    """Selecciona contexto bajo presupuesto de entrada + reserva de salida.

    El presupuesto se aplica sobre el texto renderizado real, incluidos los
    encabezados ``## ...`` y los separadores entre secciones.
    """

    def __init__(
        self,
        total_budget_tokens: int,
        reserve_output_tokens: int,
    ):
        """Configura presupuesto total y reserva para la respuesta del modelo."""
        if total_budget_tokens <= 0 or reserve_output_tokens < 0:
            raise ValueError("El total debe ser positivo y la reserva no negativa")
        if (
            total_budget_tokens
            <= reserve_output_tokens
        ):
            raise ValueError(
                "El presupuesto total debe "
                "superar la reserva de salida"
            )

        self.total_budget_tokens = (total_budget_tokens)
        self.reserve_output_tokens = (reserve_output_tokens)
        self.items: list[ContextItem] = []

    @property
    def input_budget_tokens(self) -> int:
        """Devuelve los tokens disponibles para el contexto de entrada."""
        return (
            self.total_budget_tokens
            - self.reserve_output_tokens
        )

    def add(
        self,
        name: str,
        text: str,
        *,
        priority: int,
        required: bool = False,
        allow_truncate: bool = True,
        max_tokens: int | None = None,
    ):
        """Agrega una sección candidata al contexto.

        Los bloques vacíos se ignoran. La función no decide todavía si la
        sección entra: esa decisión ocurre en ``build`` cuando conocemos el
        presupuesto y el orden global.
        """
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens debe ser positivo")
        if not str(text).strip():
            return

        self.items.append(
            ContextItem(
                name=name,
                text=str(text).strip(),
                priority=priority,
                required=required,
                allow_truncate=(allow_truncate),
                max_tokens=max_tokens,
            )
        )

    def _prepare_text(
        self,
        item: ContextItem,
    ) -> tuple[str, bool]:
        """Aplica el límite local de una sección antes del presupuesto global."""
        text = item.text
        truncated = False

        if (
            item.max_tokens
            is not None
            and estimate_tokens(text)
            > item.max_tokens
        ):
            if not item.allow_truncate:
                return (
                    text,
                    False,
                )

            text = truncate_to_tokens(
                text,
                item.max_tokens,
            )
            truncated = True

        return (
            text,
            truncated,
        )

    @staticmethod
    def _render_block(
        name: str,
        text: str,
    ) -> str:
        """Renderiza exactamente una sección como aparecerá en el prompt."""
        return (
            f"## {name}\n"
            f"{text}"
        )

    def _fit_optional_text(
        self,
        *,
        name: str,
        text: str,
        rendered_parts: list[str],
    ) -> str:
        """Busca el fragmento más largo que todavía entra en el presupuesto.

        Usa búsqueda binaria sobre el límite estimado de tokens del contenido.
        De este modo también quedan contabilizados el encabezado y los
        separadores del prompt.
        """
        low = 0
        high = estimate_tokens(text)
        best = ""

        while low <= high:
            mid = (low + high) // 2

            fragment = (
                truncate_to_tokens(
                    text,
                    mid,
                )
                if mid > 0
                else ""
            )

            if not fragment:
                projected_parts = list(rendered_parts)
            else:
                projected_parts = [
                    *rendered_parts,
                    self._render_block(
                        name,
                        fragment,
                    ),
                ]

            projected = "\n\n".join(projected_parts)

            if (
                estimate_tokens(projected)
                <= self.input_budget_tokens
            ):
                best = fragment
                low = mid + 1
            else:
                high = mid - 1

        return best

    def build(self) -> dict[str, Any]:
        """Construye el contexto final respetando prioridad y presupuesto.

        Returns
        -------
        dict
            Incluye texto renderizado, secciones incluidas/descartadas,
            presupuesto usado, reserva de salida y tokens remanentes.

        Las secciones obligatorias se evalúan primero. Si una obligatoria no
        entra, fallamos de forma explícita en vez de producir un prompt que
        viola el contrato de contexto.
        """
        included = []
        dropped = []
        rendered_parts: list[str] = []

        ordered = sorted(
            self.items,
            key=lambda item: (
                not item.required,
                -item.priority,
                item.name,
            ),
        )

        for item in ordered:
            if (item.max_tokens is not None
                    and estimate_tokens(item.text) > item.max_tokens
                    and not item.allow_truncate):
                if item.required:
                    raise RuntimeError(f"Límite local insuficiente: {item.name}")
                dropped.append({"name": item.name, "reason": "section_limit",
                                "priority": item.priority})
                continue
            text, pre_truncated = (self._prepare_text(item))

            block = self._render_block(
                item.name,
                text,
            )

            projected_parts = [
                *rendered_parts,
                block,
            ]

            projected = "\n\n".join(projected_parts)

            previous_text = ("\n\n".join(rendered_parts))

            previous_tokens = (estimate_tokens(previous_text))

            projected_tokens = (estimate_tokens(projected))

            if (
                projected_tokens
                <= self.input_budget_tokens
            ):
                rendered_parts = (projected_parts)

                included.append({
                    "name":
                        item.name,
                    "text":
                        text,
                    # ``tokens`` mide el costo incremental real
                    # de esta sección ya renderizada.
                    "tokens":
                        projected_tokens
                        - previous_tokens,
                    "content_tokens":
                        estimate_tokens(text),
                    "priority":
                        item.priority,
                    "required":
                        item.required,
                    "truncated":
                        pre_truncated,
                })
                continue

            if item.required:
                raise RuntimeError(
                    "Context budget insuficiente "
                    "para sección obligatoria: "
                    f"{item.name}"
                )

            if item.allow_truncate:
                fragment = (
                    self._fit_optional_text(
                        name=item.name,
                        text=text,
                        rendered_parts=(rendered_parts),
                    )
                )

                # Un fragmento diminuto suele aportar ruido. Pedimos un mínimo
                # de 24 tokens de contenido para incluir una sección parcial.
                if (
                    fragment
                    and estimate_tokens(fragment)
                    >= 24
                ):
                    fitted_block = (
                        self._render_block(
                            item.name,
                            fragment,
                        )
                    )

                    new_parts = [
                        *rendered_parts,
                        fitted_block,
                    ]

                    new_text = "\n\n".join(new_parts)

                    new_tokens = (estimate_tokens(new_text))

                    included.append({
                        "name":
                            item.name,
                        "text":
                            fragment,
                        "tokens":
                            new_tokens
                            - previous_tokens,
                        "content_tokens":
                            estimate_tokens(fragment),
                        "priority":
                            item.priority,
                        "required":
                            item.required,
                        "truncated":
                            True,
                    })

                    rendered_parts = (new_parts)
                    continue

            dropped.append({
                "name":
                    item.name,
                "reason":
                    "budget",
                "estimated_tokens":
                    projected_tokens
                    - previous_tokens,
                "priority":
                    item.priority,
            })

        rendered = "\n\n".join(rendered_parts)

        used_input_tokens = (estimate_tokens(rendered))

        return {
            "text":
                rendered,
            "items":
                included,
            "dropped":
                dropped,
            "used_input_tokens":
                used_input_tokens,
            "input_budget_tokens":
                self.input_budget_tokens,
            "reserved_output_tokens":
                self.reserve_output_tokens,
            "remaining_input_tokens":
                self.input_budget_tokens
                - used_input_tokens,
        }


def compact_conversation(
    messages: list[dict],
    *,
    keep_last: int = 4,
    summary_item_chars: int = 180,
) -> dict[str, Any]:
    """Compacta mensajes viejos y conserva una ventana reciente.

    Parameters
    ----------
    messages:
        Historial ordenado de mensajes.
    keep_last:
        Cantidad de mensajes recientes que permanecen sin resumir.
    summary_item_chars:
        Máximo de caracteres conservados por mensaje antiguo.

    Returns
    -------
    dict
        ``summary`` con el historial antiguo compactado, ``recent`` con la
        ventana intacta y ``compacted_count`` para observabilidad.

    Esta implementación es deliberadamente simple y determinista. En sistemas
    reales, el resumen también debería evaluarse para detectar pérdida o
    distorsión de hechos críticos.
    """
    if keep_last < 0 or summary_item_chars <= 0:
        raise ValueError("keep_last debe ser >= 0 y summary_item_chars > 0")

    if len(messages) <= keep_last:
        return {
            "summary": "",
            "recent": list(messages),
            "compacted_count": 0,
        }

    # En Python, -0 equivale a 0: tratamos este caso explícitamente.
    old = messages[:-keep_last] if keep_last else messages
    recent = messages[-keep_last:] if keep_last else []

    summary_lines = []

    for message in old:
        kind = message.get(
            "kind",
            "context",
        )

        role = message.get(
            "role",
            "unknown",
        )

        content = " ".join(
            str(
                message.get(
                    "content",
                    "",
                )
            ).split()
        )

        content = content[:summary_item_chars]

        summary_lines.append(
            f"- [{kind}] "
            f"{role}: "
            f"{content}"
        )

    return {
        "summary":
            "Resumen acumulado:\n"
            + "\n".join(summary_lines),
        "recent":
            recent,
        "compacted_count":
            len(old),
    }

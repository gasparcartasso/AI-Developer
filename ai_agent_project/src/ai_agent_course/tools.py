from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from time import perf_counter, sleep
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

# Clasificación de capacidades y permisos del runtime:
#
# ToolKind describe qué tipo de operación realiza una Tool.
# La Policy usa esta categoría para aplicar mínimo privilegio:
#
# - read: consulta información sin modificar estado.
# - compute: realiza cálculos o transformaciones.
# - local_write: genera artefactos locales, como reportes.
# - draft: prepara una acción, pero todavía no la ejecuta.
# - external_effect: interactúa con un servicio externo de forma controlada.
# - write: modifica estado de manera efectiva.
#
# Role representa el nivel de acceso del contexto confiable de ejecución.
# No lo decide el modelo ni el contenido externo.
#
# Literal restringe los valores válidos y VALID_TOOL_KINDS permite
# validar categorías de Tools en tiempo de ejecución.

ToolKind = Literal[
    "read",
    "compute",
    "local_write",
    "draft",
    "external_effect",
    "write",
]
Role = Literal["viewer", "support_agent", "admin"]
VALID_TOOL_KINDS = {
    "read", "compute", "local_write", "draft", "external_effect", "write"
}


class ExecutionContext(BaseModel):
    """Contexto confiable construido por la aplicación, nunca por contenido recuperado."""

    user_id: str
    role: Role
    workspace_ids: list[str] = Field(default_factory=list)
    allow_drafts: bool = False
    allow_external_writes: bool = False
    run_id: str = "run-local"


class ToolCall(BaseModel):
    """Solicitud estructurada propuesta por el modelo/router: Tool + argumentos."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Salida estable que el runtime convierte en una Observation."""

    ok: bool
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetryableToolError(RuntimeError):
    """Marca explícitamente un fallo transitorio que sí admite retry automático."""


AuthorizationGuard = Callable[[ExecutionContext, dict[str, Any]], str | None]


class Tool:
    """Contrato runtime-facing y model-facing de una capacidad del agente."""

    def __init__(
        self,
        name: str,
        description: str,
        kind: ToolKind,
        fn: Callable[[dict], ToolResult],
        args_model: type[BaseModel] | None = None,
        timeout_s: float = 3.0,
        retries: int = 0,
        *,
        backoff_s: float = 0.05,
        idempotent: bool = True,
        visible_roles: set[Role] | None = None,
        authorize: AuthorizationGuard | None = None,
    ):
        """Configura una Tool y sus controles operativos.

        Args:
            name: Identificador estable usado en Tool Calls.
            description: Instrucción model-facing sobre propósito y uso.
            kind: Categoría usada por Policy para mínimo privilegio.
            fn: Handler Python que ejecuta la capacidad.
            args_model: Schema Pydantic de entrada; `None` acepta un dict libre.
            timeout_s: Máximo de espera del runtime para una ejecución.
            retries: Cantidad de reintentos para `RetryableToolError`.
            backoff_s: Espera base entre reintentos.
            idempotent: Indica si repetir la operación es seguro.
            visible_roles: Roles que pueden ver la Tool; `None` no agrega filtro.
            authorize: Guard opcional para autorizar una llamada concreta.

        Returns:
            None. Deja configurado el contrato.

        Raises:
            ValueError: Si la configuración operativa es insegura o inválida.
        """
        if kind not in VALID_TOOL_KINDS:
            raise ValueError(f"{name}: kind desconocido: {kind}.")
        if timeout_s <= 0:
            raise ValueError(f"{name}: timeout_s debe ser mayor que 0.")
        if retries < 0:
            raise ValueError(f"{name}: retries no puede ser negativo.")
        if backoff_s < 0:
            raise ValueError(f"{name}: backoff_s no puede ser negativo.")
        if retries > 0 and not idempotent:
            raise ValueError(
                f"{name}: una Tool no idempotente no puede tener retries automáticos "
                "sin una estrategia explícita contra duplicados."
            )

        self.name = name
        self.description = description
        self.kind = kind
        self.fn = fn
        self.args_model = args_model
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.idempotent = idempotent
        self.visible_roles = visible_roles
        self.authorize_guard = authorize

    @property
    def risk(self) -> str:
        """Devuelve la categoría de riesgo/capacidad usada por la Policy."""
        return self.kind

    def function_declaration(self) -> dict:
        """Construye el contrato model-facing para Function Calling.

        Returns:
            Diccionario con `name`, `description` y JSON Schema de parámetros.

        El handler, los permisos y los controles operativos no se exponen al modelo:
        siguen siendo responsabilidad de la aplicación.
        """
        parameters = (
            self.args_model.model_json_schema()
            if self.args_model
            else {"type": "object", "properties": {}}
        )

        # Los `title` de Pydantic aportan poco al modelo y agregan ruido visual.
        parameters.pop("title", None)
        for prop in parameters.get("properties", {}).values():
            prop.pop("title", None)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }


class ToolPolicy:
    """Decide qué Tools son visibles y autoriza llamadas concretas."""

    KINDS_BY_ROLE = {
        "viewer": {"read", "compute", "local_write"},
        "support_agent": {
            "read",
            "compute",
            "local_write",
            "draft",
            "external_effect",
        },
        "admin": {
            "read",
            "compute",
            "local_write",
            "draft",
            "external_effect",
            "write",
        },
    }

    def allowed_tools(
        self,
        registry: "ToolRegistry",
        context: ExecutionContext,
    ) -> list[Tool]:
        """Filtra las Tools visibles para un contexto.

        Args:
            registry: Registry completo de capacidades.
            context: Contexto confiable de usuario, rol y flags.

        Returns:
            Lista de Tools que el modelo puede ver en ese contexto.
        """
        return self.allowed_tools_from_list(
            list(registry.tools.values()),
            context,
        )

    def authorize(
        self,
        context: ExecutionContext,
        tool: Tool,
        args: dict[str, Any],
    ) -> str | None:
        """Autoriza una llamada concreta después de validar argumentos.

        Args:
            context: Contexto confiable de ejecución.
            tool: Tool ya resuelta por el Registry.
            args: Argumentos ya validados por Pydantic.

        Returns:
            `None` si la llamada está autorizada; texto explicativo si se rechaza.
        """
        visible = {
            item.name
            for item in self.allowed_tools_from_list([tool], context)
        }
        if tool.name not in visible:
            return "La Tool no está habilitada para este contexto."

        if tool.authorize_guard is not None:
            return tool.authorize_guard(context, args)

        return None

    def allowed_tools_from_list(
        self,
        tools: list[Tool],
        context: ExecutionContext,
    ) -> list[Tool]:
        """Aplica reglas de rol, flags y visibilidad a una lista de Tools.

        Args:
            tools: Capacidades candidatas.
            context: Contexto confiable de ejecución.

        Returns:
            Subconjunto visible según mínimo privilegio.
        """
        allowed_kinds = set(self.KINDS_BY_ROLE[context.role])

        if not context.allow_drafts:
            allowed_kinds.discard("draft")

        if not context.allow_external_writes:
            allowed_kinds.discard("external_effect")
            allowed_kinds.discard("write")

        return [
            tool
            for tool in tools
            if tool.kind in allowed_kinds
            and (
                tool.visible_roles is None
                or context.role in tool.visible_roles
            )
        ]


class ToolRegistry:
    """Registry con availability, validation, authorization, timeout y retry."""

    DEFAULT_ALLOWED_KINDS: set[ToolKind] = {"read", "compute", "local_write"}

    def __init__(self):
        """Crea un registry vacío indexado por nombre de Tool."""
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Registra una Tool sin permitir nombres duplicados.

        Args:
            tool: Contrato completo que se quiere incorporar.

        Returns:
            None.

        Raises:
            ValueError: Si el nombre ya estaba registrado.
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool duplicada: {tool.name}")
        self.tools[tool.name] = tool

    def available_tools(
        self,
        allowed_kinds: set[str] | None = None,
        *,
        context: ExecutionContext | None = None,
        policy: ToolPolicy | None = None,
    ) -> list[Tool]:
        """Devuelve las Tools visibles por lista de kinds o por Policy.

        Args:
            allowed_kinds: Filtro simple para demos sin contexto. Un set vacío
                significa explícitamente "ninguna Tool".
            context: Contexto confiable, usado junto con `policy`.
            policy: Policy que calcula mínimo privilegio.

        Returns:
            Lista de Tools disponibles para el modo elegido.
        """
        if (context is None) != (policy is None):
            raise ValueError("context y policy deben proporcionarse juntos.")

        if context is not None and policy is not None:
            return policy.allowed_tools(self, context)

        allowed = (
            self.DEFAULT_ALLOWED_KINDS
            if allowed_kinds is None
            else allowed_kinds
        )
        return [
            tool
            for tool in self.tools.values()
            if tool.kind in allowed
        ]

    def function_declarations(
        self,
        allowed_kinds: set[str] | None = None,
        *,
        context: ExecutionContext | None = None,
        policy: ToolPolicy | None = None,
    ) -> list[dict]:
        """Convierte las Tools visibles en declaraciones para el modelo.

        Returns:
            Lista de contratos model-facing; nunca incluye handlers ni permisos.
        """
        return [
            tool.function_declaration()
            for tool in self.available_tools(
                allowed_kinds,
                context=context,
                policy=policy,
            )
        ]

    def run(
        self,
        call: ToolCall,
        allowed_kinds: set[str] | None = None,
        *,
        context: ExecutionContext | None = None,
        policy: ToolPolicy | None = None,
    ) -> ToolResult:
        """Ejecuta un Tool Call atravesando todas las capas del runtime.

        Args:
            call: Nombre y argumentos propuestos por el agente.
            allowed_kinds: Filtro simple para demos sin Policy.
            context: Contexto confiable de ejecución.
            policy: Reglas de availability y authorization.

        Returns:
            `ToolResult` exitoso o un error controlado. Las excepciones esperables
            no escapan al Agent Loop.

        Orden de controles:
            Registry → Availability → Validation → Authorization → Execution.
        """
        if (context is None) != (policy is None):
            raise ValueError("context y policy deben proporcionarse juntos.")

        # 1. Registry: ¿existe?
        tool = self.tools.get(call.tool_name)
        if tool is None:
            return ToolResult(
                ok=False,
                content=f"Tool desconocida: {call.tool_name}",
                metadata={"error_type": "unknown_tool"},
            )

        # 2. Availability: ¿se expone en este contexto?
        if context is not None and policy is not None:
            visible = {
                item.name
                for item in policy.allowed_tools(self, context)
            }
            if tool.name not in visible:
                return ToolResult(
                    ok=False,
                    content=f"Tool no disponible en este contexto: {tool.name}",
                    metadata={
                        "error_type": "unavailable_tool",
                        "role": context.role,
                        "kind": tool.kind,
                    },
                )
        else:
            allowed = (
                self.DEFAULT_ALLOWED_KINDS
                if allowed_kinds is None
                else allowed_kinds
            )
            if tool.kind not in allowed:
                return ToolResult(
                    ok=False,
                    content=f"Tool bloqueada por política: {tool.name}",
                    metadata={
                        "error_type": "forbidden_tool",
                        "kind": tool.kind,
                    },
                )

        # 3. Validation: ¿los argumentos tienen forma válida?
        try:
            args = (
                tool.args_model.model_validate(call.arguments).model_dump()
                if tool.args_model
                else dict(call.arguments)
            )
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                content="Argumentos inválidos",
                metadata={
                    "error_type": "validation_error",
                    "details": exc.errors(include_url=False),
                },
            )

        # 4. Authorization: ¿esta llamada concreta está permitida?
        if context is not None and policy is not None:
            denial = policy.authorize(context, tool, args)
            if denial:
                return ToolResult(
                    ok=False,
                    content=denial,
                    metadata={
                        "error_type": "authorization_error",
                        "tool": tool.name,
                    },
                )

        # 5. Execution. Solo `RetryableToolError` habilita un nuevo intento.
        last_retryable_error: RetryableToolError | None = None

        for attempt in range(1, tool.retries + 2):
            if attempt > 1:
                # Backoff exponencial mínimo para recordar C3.
                sleep(tool.backoff_s * (2 ** (attempt - 2)))

            executor = ThreadPoolExecutor(max_workers=1)
            started = perf_counter()

            try:
                future = executor.submit(tool.fn, args)
                raw_result = future.result(timeout=tool.timeout_s)
                elapsed_ms = (perf_counter() - started) * 1000

                # El output contract también se valida. Un handler que devuelve
                # otra forma no debe contaminar el resto del loop.
                try:
                    result = ToolResult.model_validate(raw_result)
                except ValidationError as exc:
                    return ToolResult(
                        ok=False,
                        content=f"Salida inválida de {tool.name}",
                        metadata={
                            "error_type": "output_validation_error",
                            "tool": tool.name,
                            "details": exc.errors(include_url=False),
                        },
                    )

                result.metadata = {
                    **result.metadata,
                    "tool": tool.name,
                    "kind": tool.kind,
                    "attempt": attempt,
                    "latency_ms": round(elapsed_ms, 2),
                    "idempotent": tool.idempotent,
                }
                return result

            except FuturesTimeout:
                # Cancelar la espera NO garantiza detener un thread ya iniciado.
                # El handler debe ser idempotente o diseñarse con cancelación cooperativa.
                # Por eso un timeout nunca se reintenta automáticamente acá.
                future.cancel()
                return ToolResult(
                    ok=False,
                    content=f"Timeout ejecutando {tool.name}",
                    metadata={
                        "error_type": "timeout",
                        "tool": tool.name,
                        "timeout_s": tool.timeout_s,
                        "attempt": attempt,
                    },
                )

            except RetryableToolError as exc:
                last_retryable_error = exc
                if attempt > tool.retries:
                    break

            except Exception as exc:
                # Un error no marcado como transitorio no se repite: podría ser
                # determinístico o esconder un bug del handler.
                return ToolResult(
                    ok=False,
                    content=f"Error controlado: {exc}",
                    metadata={
                        "error_type": type(exc).__name__,
                        "tool": tool.name,
                        "attempt": attempt,
                        "retryable": False,
                    },
                )

            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        return ToolResult(
            ok=False,
            content=f"Se agotaron los retries: {last_retryable_error}",
            metadata={
                "error_type": "retry_exhausted",
                "exception_type": type(last_retryable_error).__name__,
                "tool": tool.name,
                "attempts": tool.retries + 1,
            },
        )

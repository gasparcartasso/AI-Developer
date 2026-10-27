from __future__ import annotations

import asyncio
import operator
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class WorkerTask(BaseModel):
    """Contrato de entrada: una tarea, un rol autorizado y dependencias explícitas."""
    task_id: str = Field(min_length=2)
    role: str = Field(min_length=2)
    objective: str = Field(min_length=10, max_length=1000)
    required_inputs: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    allowed_context: dict[str, str] = Field(default_factory=dict)
    max_steps: int = Field(default=3, ge=1, le=5)


class Evidence(BaseModel):
    """Proposición identificable y lugar concreto de su fuente."""
    claim: str = Field(min_length=1)
    source: str = Field(min_length=1)
    locator: str | None = None


class WorkerOutput(BaseModel):
    """Resultado normalizado; 'partial' y 'failed' son salidas válidas."""
    task_id: str
    role: str
    status: Literal["success", "partial", "failed"]
    summary: str
    evidence: list[Evidence] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    error: str | None = None
    confidence: float = Field(default=0, ge=0, le=1)
    latency_ms: float = Field(default=0, ge=0)


# Firma esperada para los workers del registry:
# recibe (tarea, outputs de sus dependencias) y retorna un WorkerOutput async.
WorkerFn = Callable[[WorkerTask, list[WorkerOutput]], Awaitable[WorkerOutput]]


def validate_plan(tasks: list[WorkerTask], workers: dict[str, WorkerFn],
                  *, max_tasks: int = 8, max_concurrency: int = 2) -> None:
    """Valida TODO plan — incluso uno generado con salida estructurada de un LLM.

    Comprueba identidad única, roles existentes, dependencias conocidas, ciclos,
    presupuesto global y ausencia de autorreferencias. No ejecuta tools.
    Raises: ValueError si alguna regla del runtime se incumple.
    """
    if not 1 <= len(tasks) <= max_tasks:
        raise ValueError(f"Cantidad de tareas fuera de presupuesto: {len(tasks)}")
    if not 1 <= max_concurrency <= 4:
        raise ValueError("max_concurrency debe estar entre 1 y 4")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task_id duplicados")
    known = set(ids)
    for task in tasks:
        if task.role not in workers:
            raise ValueError(f"Rol no permitido: {task.role}")
        if len(task.depends_on) != len(set(task.depends_on)):
            raise ValueError(f"Dependencia duplicada: {task.task_id}")
        if task.task_id in task.depends_on:
            raise ValueError(f"Dependencia reflexiva: {task.task_id}")
        if not set(task.depends_on) <= known:
            raise ValueError(f"Dependencia inexistente: {task.task_id}")

    # Ordenamiento topológico mínimo: una ola avanza sólo con dependencias conocidas.
    pending = {task.task_id: task for task in tasks}
    completed: set[str] = set()
    while pending:
        # Fan-out lógico: misma ola para tareas independientes.
        ready = [tid for tid, task in pending.items()
                 if set(task.depends_on) <= completed]
        if not ready:
            raise ValueError("Dependencias cíclicas")
        completed.update(ready)
        for tid in ready:
            pending.pop(tid)


async def _execute(task: WorkerTask, deps: list[WorkerOutput],
                   workers: dict[str, WorkerFn], semaphore: asyncio.Semaphore,
                   timeout_s: float) -> WorkerOutput:
    """Invoca UN worker, impone timeout y valida la identidad del resultado.

    El error público conserva sólo el TIPO de excepción para no filtrar secretos.
    En una app real el stack detallado va a logs internos protegidos.
    """
    started = time.perf_counter()
    try:
        # El semáforo compartido limita llamadas simultáneas a tools/modelos.
        async with semaphore:
            output = await asyncio.wait_for(
                workers[task.role](task, deps), timeout=timeout_s)
        output = WorkerOutput.model_validate(output)
        if output.task_id != task.task_id or output.role != task.role:
            raise ValueError("El worker devolvió identidad no coincidente")
    except asyncio.TimeoutError:
        output = WorkerOutput(task_id=task.task_id, role=task.role,
                              status="failed", summary="Worker fuera de tiempo",
                              error="timeout")
    except Exception as exc:
        output = WorkerOutput(task_id=task.task_id, role=task.role,
                              status="failed", summary="Falla controlada del worker",
                              error=type(exc).__name__)
    output.latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return output


def _result(tasks: list[WorkerTask], outputs: dict[str, WorkerOutput],
            trace: list[dict], elapsed_ms: float, engine: str) -> dict:
    """Consolida por ID — no depende del orden de llegada del fan-out."""
    # Convención didáctica: la última tarea del plan es el consolidator final.
    # No sirve para DAG arbitrarios con múltiples nodos terminales.
    target = outputs[tasks[-1].task_id]
    return {"outputs": outputs, "trace": trace,
            "final": target.summary, "status": target.status,
            "total_latency_ms": round(elapsed_ms, 2), "engine": engine}


async def run_supervisor_workers(
    tasks: list[WorkerTask], workers: dict[str, WorkerFn],
    max_concurrency: int = 2, timeout_s: float = 10.0,
) -> dict:
    """Versión Python por olas, útil para comparar contra el grafo de LangGraph.

    Args: tasks DAG pequeño; workers mapa cerrado de roles; max_concurrency
    límite compartido; timeout_s máximo por llamada.
    Returns: outputs tipados, traza causal y estado final.
    """
    validate_plan(tasks, workers, max_concurrency=max_concurrency)
    if timeout_s <= 0:
        raise ValueError("timeout_s debe ser positivo")
    started = time.perf_counter()
    pending = {task.task_id: task for task in tasks}
    outputs: dict[str, WorkerOutput] = {}
    trace: list[dict] = []
    semaphore = asyncio.Semaphore(max_concurrency)
    wave = 0
    while pending:
        ready = [task for task in pending.values()
                 if all(dep in outputs for dep in task.depends_on)]
        wave += 1
        results = await asyncio.gather(*[
            _execute(task, [outputs[d] for d in task.depends_on],
                     workers, semaphore, timeout_s) for task in ready
        ])
        for task, output in zip(ready, results, strict=True):
            outputs[task.task_id] = output
            pending.pop(task.task_id)
            trace.append({"event": "worker_completed", "wave": wave,
                          "role": output.role, "task_id": output.task_id,
                          "status": output.status, "latency_ms": output.latency_ms,
                          "dependencies": task.depends_on})
    return _result(tasks, outputs, trace,
                   (time.perf_counter() - started) * 1000,
                   engine="python_waves")


class GraphState(TypedDict, total=False):
    """State global. operator.add acumula outputs de ramas simultáneas."""
    tasks: list[dict]
    remaining: list[str]
    ready: list[str]
    # El reducer CONCATENA actualizaciones de ramas; no deduplica ni interpreta.
    worker_outputs: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
    final: str
    status: str


async def run_supervisor_workers_langgraph(
    tasks: list[WorkerTask], workers: dict[str, WorkerFn],
    max_concurrency: int = 2, timeout_s: float = 10.0,
) -> dict:
    """Ejecuta el mismo DAG con StateGraph + Send + reducer y barrera por ola.

    Necesita langgraph instalado; no simula el framework si falta. El fan-in
    actualiza el State global una sola vez después de recibir todas las ramas.
    """
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send

    validate_plan(tasks, workers, max_concurrency=max_concurrency)
    if timeout_s <= 0:
        raise ValueError("timeout_s debe ser positivo")
    started = time.perf_counter()
    task_by_id = {t.task_id: t for t in tasks}
    semaphore = asyncio.Semaphore(max_concurrency)  # límite global real

    def plan_wave(state: GraphState) -> dict:
        """Elige ids listos para ejecutar. Args: estado global. Returns: ready."""
        already = {o["task_id"] for o in state.get("worker_outputs", [])}
        ready = [tid for tid in state["remaining"]
                 if set(task_by_id[tid].depends_on) <= already]
        if state["remaining"] and not ready:
            raise ValueError("No hay tareas ejecutables")
        return {"ready": ready}

    def dispatch(state: GraphState):
        """Routing condicional: crea Send por tarea o deriva a consolidación."""
        if not state["ready"]:
            return "consolidate"
        results = {o["task_id"]: o for o in state.get("worker_outputs", [])}
        # Send permite pasar a cada worker SÓLO tarea y dependencias autorizadas.
        return [Send("worker", {"task": task_by_id[tid].model_dump(),
                               "dependencies": [results[d] for d in task_by_id[tid].depends_on]})
                for tid in state["ready"]]

    async def worker_node(payload: dict) -> dict:
        """Ejecuta una tarea y actualiza listas; el reducer fusiona las ramas."""
        task = WorkerTask.model_validate(payload["task"])
        deps = [WorkerOutput.model_validate(o) for o in payload["dependencies"]]
        out = await _execute(task, deps, workers, semaphore, timeout_s)
        return {"worker_outputs": [out.model_dump()],
                "trace": [{"event": "worker_completed", "role": out.role,
                           "task_id": out.task_id, "status": out.status,
                           "latency_ms": out.latency_ms,
                           "dependencies": task.depends_on}]}

    def advance(state: GraphState) -> dict:
        """Quita del conjunto pendiente las tareas que ya devolvieron salida."""
        completed = {o["task_id"] for o in state.get("worker_outputs", [])}
        return {"remaining": [tid for tid in state["remaining"] if tid not in completed]}

    def consolidate(state: GraphState) -> dict:
        """Verifica fan-in completo y toma el resultado del último worker."""
        items = state.get("worker_outputs", [])
        outputs = {o["task_id"]: o for o in items}
        if len(items) != len(outputs) or len(outputs) != len(tasks):
            raise ValueError("Fan-in incompleto o task_id duplicado")
        final = outputs[tasks[-1].task_id]
        return {"final": final["summary"], "status": final["status"]}

    graph = StateGraph(GraphState)
    graph.add_node("plan_wave", plan_wave)
    graph.add_node("worker", worker_node)
    graph.add_node("advance", advance)
    graph.add_node("consolidate", consolidate)
    graph.add_edge(START, "plan_wave")
    graph.add_conditional_edges("plan_wave", dispatch)
    graph.add_edge("worker", "advance")
    graph.add_edge("advance", "plan_wave")
    graph.add_edge("consolidate", END)
    # Cada wave espera actualizaciones de las ramas antes de calcular qué tareas faltan.
    app = graph.compile()
    # El StateGraph se ejecuta de verdad; async evita bloquear la sesión.
    state = await app.ainvoke(
        {"tasks": [t.model_dump() for t in tasks],
         "remaining": [t.task_id for t in tasks], "ready": [],
         "worker_outputs": [], "trace": []},
        config={"max_concurrency": max_concurrency, "recursion_limit": 40},
    )
    outs = {o["task_id"]: WorkerOutput.model_validate(o)
            for o in state["worker_outputs"]}
    return _result(tasks, outs, state["trace"],
                   (time.perf_counter() - started) * 1000,
                   engine="langgraph_send_reducer")

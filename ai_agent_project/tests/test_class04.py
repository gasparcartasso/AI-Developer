import asyncio

from ai_agent_course.local_models import SimulatedAgentProvider
from ai_agent_course.routing import choose_provider_for_task


def test_local_route_for_sensitive_simple_task():
    decision = choose_provider_for_task(
        text="El mensaje incluye un DNI",
        needs_tools=False,
        complexity="low",
        local_available=True,
        cloud_available=True,
    )
    assert decision.provider == "local"


def test_sensitive_tool_task_never_escalates_silently():
    decision = choose_provider_for_task(
        text="Buscar el legajo asociado a este DNI",
        needs_tools=True,
        complexity="medium",
        local_available=True,
        cloud_available=True,
    )
    assert decision.provider == "local"
    assert decision.requires_human_review is True


def test_cloud_route_for_tools():
    decision = choose_provider_for_task(
        text="Buscar una política",
        needs_tools=True,
        complexity="medium",
        local_available=True,
        cloud_available=True,
    )
    assert decision.provider == "cloud"


def test_provider_returns_common_contract():
    provider = SimulatedAgentProvider(
        name="test",
        model="test-model",
        latency_ms=1,
        supports_tools=True,
        structured_reliability=1.0,
    )
    result = asyncio.run(provider.agenerate("Devolvé SOLO JSON con category y confidence"))
    assert result.provider == "test"
    assert result.model == "test-model"
    assert result.text


def test_native_tool_call_has_structured_arguments():
    provider = SimulatedAgentProvider(
        name="test",
        model="test-model",
        latency_ms=1,
        supports_tools=True,
        structured_reliability=1.0,
    )
    tool = {
        "type": "function",
        "function": {
            "name": "buscar_politica",
            "description": "Busca una política.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    result = asyncio.run(
        provider.achat_with_tools("Buscá vacaciones", tools=[tool])
    )
    call = result["tool_calls"][0]["function"]
    assert call["name"] == "buscar_politica"
    assert call["arguments"]["query"] == "vacaciones"

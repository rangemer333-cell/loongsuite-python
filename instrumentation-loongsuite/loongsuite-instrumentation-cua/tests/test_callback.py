# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the ArmsCuaCallback.

These tests exercise the callback against a stub agent (no real CUA / LLM
calls) and verify the resulting span tree, attribute keys, and parent /
child relationships follow the ARMS GenAI semantic conventions.
"""

import asyncio
import types
from uuid import UUID

import pytest

from opentelemetry.instrumentation.cua.callback import ArmsCuaCallback
from opentelemetry.instrumentation.cua.utils import (
    action_description,
    compute_agent_id,
    extract_finish_reason,
    extract_provider_from_model,
)


class StubAgent:
    def __init__(self, model="anthropic/claude-sonnet-4-5-20250929", instructions="You are a helpful assistant"):
        self.model = model
        self.instructions = instructions
        self.callbacks = []


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _span_kind(span):
    attrs = dict(span.attributes or {})
    return attrs.get("gen_ai.span.kind")


def _attr(span, key):
    attrs = dict(span.attributes or {})
    return attrs.get(key)


def _parent_id(span):
    parent = span.parent
    return getattr(parent, "span_id", None) if parent else None


def _span_id(span):
    return span.context.span_id


def test_provider_inference():
    assert extract_provider_from_model("anthropic/claude-sonnet-4-5") == "anthropic"
    assert extract_provider_from_model("openai/gpt-4o") == "openai"
    assert extract_provider_from_model("claude-sonnet-4-5") == "anthropic"
    assert extract_provider_from_model("gpt-4o-mini") == "openai"
    assert extract_provider_from_model("gemini-1.5-pro") == "google"
    assert extract_provider_from_model("") == "unknown"


def test_compute_agent_id_stable():
    a = compute_agent_id("anthropic/claude", "do thing")
    b = compute_agent_id("anthropic/claude", "do thing")
    c = compute_agent_id("anthropic/claude", "do other")
    assert a == b
    assert a != c
    assert a is not None and len(a) == 16


def test_action_description_known_and_unknown():
    assert action_description("click") == "Click at coordinates"
    assert action_description("type") == "Type text"
    assert action_description("nonexistent_action") == "Computer action"


def test_extract_finish_reason_stop_and_tool_calls():
    assert extract_finish_reason({"output": [{"role": "assistant", "content": "done"}]}) == "stop"
    assert extract_finish_reason({"output": [{"role": "assistant", "tool_calls": [{"id": "x"}]}]}) == "tool_calls"
    assert extract_finish_reason({"output": []}) == "unknown"
    assert extract_finish_reason({}) == "unknown"
    assert extract_finish_reason("not a dict") == "unknown"


def test_full_run_emits_entry_agent_step_tool(span_exporter, handler):
    agent = StubAgent()
    cb = ArmsCuaCallback(handler, agent)

    async def scenario():
        await cb.on_run_start({"model": "anthropic/claude-sonnet-4-5"}, [])
        await cb.on_llm_start([{"role": "user", "content": "hi"}])
        await cb.on_responses({}, {"output": [{"role": "assistant", "content": "ok"}]})
        await cb.on_computer_call_start(
            {"call_id": "c1", "action": {"type": "click", "x": 10, "y": 20, "button": "left"}}
        )
        await cb.on_computer_call_end(
            {"call_id": "c1"}, [{"type": "computer_call_output", "output": {"screenshot": "ok"}}]
        )
        await cb.on_function_call_start(
            {"call_id": "f1", "name": "search", "arguments": '{"query": "foo"}'}
        )
        await on_function_call_end_safe(cb, {"call_id": "f1"}, [{"type": "function_call_output", "output": "result"}])
        await cb.on_usage({"prompt_tokens": 10, "completion_tokens": 5})
        await cb.on_run_end({}, [], [])

    _run(scenario())

    spans = span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    # Expected: ENTRY, AGENT, STEP, TOOL(click), TOOL(search), in that order
    assert "enter_ai_application_system" in span_names
    assert any(n.startswith("invoke_agent") for n in span_names)
    assert "react step" in span_names
    assert "execute_tool click" in span_names
    assert "execute_tool search" in span_names

    by_name = {s.name: s for s in spans}
    entry = by_name["enter_ai_application_system"]
    agent_span = next(s for s in spans if s.name.startswith("invoke_agent"))
    step = by_name["react step"]
    click = by_name["execute_tool click"]
    search = by_name["execute_tool search"]

    # Span kinds
    assert _span_kind(entry) == "ENTRY"
    assert _span_kind(agent_span) == "AGENT"
    assert _span_kind(step) == "STEP"
    assert _span_kind(click) == "TOOL"
    assert _span_kind(search) == "TOOL"

    # Operation names
    assert _attr(entry, "gen_ai.operation.name") == "enter"
    assert _attr(agent_span, "gen_ai.operation.name") == "invoke_agent"
    assert _attr(step, "gen_ai.operation.name") == "react"
    assert _attr(click, "gen_ai.operation.name") == "execute_tool"
    assert _attr(search, "gen_ai.operation.name") == "execute_tool"

    # Parent chain: ENTRY <- AGENT <- STEP <- TOOL
    assert _parent_id(agent_span) == _span_id(entry)
    assert _parent_id(step) == _span_id(agent_span)
    assert _parent_id(click) == _span_id(step)
    assert _parent_id(search) == _span_id(step)

    # ENTRY session id is a UUID
    session_id = _attr(entry, "gen_ai.session.id")
    assert isinstance(UUID(session_id), UUID)

    # AGENT carries model + token usage
    assert _attr(agent_span, "gen_ai.agent.name") == "anthropic/claude-sonnet-4-5"
    assert _attr(agent_span, "gen_ai.provider.name") == "anthropic"
    assert _attr(agent_span, "gen_ai.usage.input_tokens") == 10
    assert _attr(agent_span, "gen_ai.usage.output_tokens") == 5

    # STEP round counter starts at 1
    assert _attr(step, "gen_ai.react.round") == 1
    assert _attr(step, "gen_ai.react.finish_reason") == "stop"

    # Computer tool: type=computer_use, action attrs
    assert _attr(click, "gen_ai.tool.type") == "computer_use"
    assert _attr(click, "gen_ai.tool.name") == "click"
    assert _attr(click, "cua.action.type") == "click"
    assert _attr(click, "cua.action.coordinate") == "(10,20)"
    assert _attr(click, "cua.action.button") == "left"

    # Function tool: type=function, name=search
    assert _attr(search, "gen_ai.tool.type") == "function"
    assert _attr(search, "gen_ai.tool.name") == "search"


def test_multiple_react_steps_increment_round(span_exporter, handler):
    agent = StubAgent()
    cb = ArmsCuaCallback(handler, agent)

    async def scenario():
        await cb.on_run_start({"model": "anthropic/claude-3"}, [])
        await cb.on_llm_start([])
        await cb.on_responses({}, {"output": [{"role": "assistant", "tool_calls": [{"id": "x"}]}]})
        await cb.on_llm_start([])
        await cb.on_responses({}, {"output": [{"role": "assistant", "content": "done"}]})
        await cb.on_run_end({}, [], [])

    _run(scenario())

    spans = span_exporter.get_finished_spans()
    steps = [s for s in spans if s.name == "react step"]
    assert len(steps) == 2
    rounds = sorted(s.attributes.get("gen_ai.react.round") for s in steps)
    assert rounds == [1, 2]
    reasons = {s.attributes.get("gen_ai.react.finish_reason") for s in steps}
    assert "tool_calls" in reasons
    assert "stop" in reasons


def test_tool_call_without_close_is_failed_on_next_step(span_exporter, handler):
    agent = StubAgent()
    cb = ArmsCuaCallback(handler, agent)

    async def scenario():
        await cb.on_run_start({"model": "anthropic/claude-3"}, [])
        await cb.on_llm_start([])
        await cb.on_responses({}, {"output": [{"role": "assistant", "content": "ok"}]})
        await cb.on_computer_call_start(
            {"call_id": "c1", "action": {"type": "screenshot"}}
        )
        # No on_computer_call_end — simulate a crash. Start next step.
        await cb.on_llm_start([])
        await cb.on_responses({}, {"output": [{"role": "assistant", "content": "done"}]})
        await cb.on_run_end({}, [], [])

    _run(scenario())

    spans = span_exporter.get_finished_spans()
    screenshot_span = next(s for s in spans if s.name == "execute_tool screenshot")
    # The orphaned tool span should be marked ERROR
    assert screenshot_span.status.is_ok is False


def test_no_content_capture_omits_arguments(span_exporter, instrument_no_content, monkeypatch):
    # Reset the capturing-mode cache and re-read env
    from opentelemetry.util.genai import types as genai_types

    # The function reads the env lazily on each call; just set env to false
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT")

    handler_obj = None
    from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler_obj = ExtendedTelemetryHandler(tracer_provider=provider)

    agent = StubAgent()
    cb = ArmsCuaCallback(handler_obj, agent)

    async def scenario():
        await cb.on_run_start({"model": "anthropic/claude-3"}, [])
        await cb.on_llm_start([])
        await cb.on_responses({}, {"output": [{"role": "assistant", "content": "ok"}]})
        await cb.on_computer_call_start(
            {"call_id": "c1", "action": {"type": "click", "x": 1, "y": 2}}
        )
        await cb.on_computer_call_end({"call_id": "c1"}, {"output": "screenshot"})
        await cb.on_run_end({}, [], [])

    _run(scenario())

    spans = exporter.get_finished_spans()
    click = next(s for s in spans if s.name == "execute_tool click")
    # When content capture is disabled, tool arguments/results are omitted
    assert "gen_ai.tool.call.arguments" not in dict(click.attributes or {})
    assert "gen_ai.tool.call.result" not in dict(click.attributes or {})


async def on_function_call_end_safe(cb, item, result):
    # Tiny helper so the test doesn't depend on the private import path
    await cb.on_function_call_end(item, result)


def test_entry_agent_carry_messages_system_instructions_framework_ttft(
    span_exporter, handler
):
    """Regression for verification non-blocking #3-#5: ENTRY/AGENT must
    serialize ``gen_ai.input.messages`` / ``gen_ai.output.messages`` /
    ``gen_ai.system_instructions`` / ``gen_ai.tool.definitions`` (when
    SPAN_ONLY is on), stamp public ``gen_ai.framework=cua``, and AGENT must
    emit ``gen_ai.response.time_to_first_token``.
    """
    import json as _json

    agent = StubAgent(instructions="You are a helpful assistant")
    agent.tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {"type": "computer", "computer": object()},
    ]
    cb = ArmsCuaCallback(handler, agent)

    async def scenario():
        await cb.on_run_start(
            {
                "model": "anthropic/claude-sonnet-4-5",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
            [],
        )
        await cb.on_llm_start([{"role": "user", "content": "hi"}])
        await cb.on_responses(
            {},
            {"output": [{"role": "assistant", "content": "ok", "finish_reason": "stop"}]},
        )
        await cb.on_run_end({}, [], [])

    _run(scenario())

    spans = span_exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    entry = by_name["enter_ai_application_system"]
    agent_span = next(s for s in spans if s.name.startswith("invoke_agent"))

    # Public gen_ai.framework on every span kind.
    for s in spans:
        assert dict(s.attributes or {}).get("gen_ai.framework") == "cua", s.name

    # ENTRY: input.messages (ENTRY handler does not emit tool.definitions
    # or system_instructions — those are only on AGENT).
    entry_attrs = dict(entry.attributes or {})
    assert "gen_ai.input.messages" in entry_attrs
    entry_input = _json.loads(entry_attrs["gen_ai.input.messages"])
    assert entry_input[0]["role"] == "user"
    assert entry_input[0]["parts"][0]["type"] == "text"
    assert entry_input[0]["parts"][0]["content"] == "hello"
    # ENTRY TTFT (was already emitted; ensure it stays a positive int)
    assert entry_attrs.get("gen_ai.response.time_to_first_token") is not None
    assert entry_attrs["gen_ai.response.time_to_first_token"] > 0

    # AGENT: input.messages, output.messages, system_instructions,
    # tool.definitions, TTFT
    agent_attrs = dict(agent_span.attributes or {})
    assert "gen_ai.input.messages" in agent_attrs
    assert _json.loads(agent_attrs["gen_ai.input.messages"])[0]["role"] == "user"
    assert "gen_ai.output.messages" in agent_attrs
    agent_output = _json.loads(agent_attrs["gen_ai.output.messages"])
    assert agent_output[0]["role"] == "assistant"
    assert agent_output[0]["finish_reason"] == "stop"
    assert "gen_ai.system_instructions" in agent_attrs
    sys_instr = _json.loads(agent_attrs["gen_ai.system_instructions"])
    assert sys_instr[0]["content"] == "You are a helpful assistant"
    assert "gen_ai.tool.definitions" in agent_attrs
    tool_defs = _json.loads(agent_attrs["gen_ai.tool.definitions"])
    assert any(td.get("name") == "get_weather" for td in tool_defs)
    assert any(td.get("type") == "computer" for td in tool_defs)
    # Regression for verification non-blocking #5: AGENT TTFT was dropped
    # because monotonic_first_token_s was set to a delta — now uses absolute
    # perf_counter reading, so the handler emits the attribute.
    assert "gen_ai.response.time_to_first_token" in agent_attrs
    assert agent_attrs["gen_ai.response.time_to_first_token"] > 0


def test_string_input_messages_serialized_on_entry_agent(
    span_exporter, handler
):
    """Regression for verification report 7ca3c1df P1.1.

    ``ComputerAgent.run("hello")`` passes a bare string as ``messages``.
    ``_messages_to_input_messages`` previously only handled lists, so the
    ENTRY/AGENT spans dropped ``gen_ai.input.messages`` even though
    ``output.messages`` was present. After the fix, a string input is
    normalized to ``[{"role": "user", "content": str}]`` and serialized on
    both ENTRY and AGENT.
    """
    import json as _json

    agent = StubAgent(instructions="You are a helpful assistant")
    cb = ArmsCuaCallback(handler, agent)

    async def scenario():
        await cb.on_run_start(
            {"model": "anthropic/claude-sonnet-4-5", "messages": "Reply with hello"},
            [],
        )
        await cb.on_llm_start([{"role": "user", "content": "Reply with hello"}])
        await cb.on_responses(
            {},
            {"output": [{"role": "assistant", "content": "hello", "finish_reason": "stop"}]},
        )
        await cb.on_run_end({}, [], [])

    _run(scenario())

    spans = span_exporter.get_finished_spans()
    entry = next(s for s in spans if s.name == "enter_ai_application_system")
    agent_span = next(s for s in spans if s.name.startswith("invoke_agent"))

    entry_attrs = dict(entry.attributes or {})
    assert "gen_ai.input.messages" in entry_attrs, "ENTRY missing gen_ai.input.messages"
    entry_input = _json.loads(entry_attrs["gen_ai.input.messages"])
    assert entry_input[0]["role"] == "user"
    assert entry_input[0]["parts"][0]["content"] == "Reply with hello"

    agent_attrs = dict(agent_span.attributes or {})
    assert "gen_ai.input.messages" in agent_attrs, "AGENT missing gen_ai.input.messages"
    agent_input = _json.loads(agent_attrs["gen_ai.input.messages"])
    assert agent_input[0]["role"] == "user"
    assert agent_input[0]["parts"][0]["content"] == "Reply with hello"
    # output.messages should also be present (regression for the "only
    # output existed" half of the bug).
    assert "gen_ai.output.messages" in agent_attrs
    assert _json.loads(agent_attrs["gen_ai.output.messages"])[0]["role"] == "assistant"


def test_framework_baggage_propagates_to_external_llm_span(
    tracer_provider, span_exporter, handler
):
    """Regression for verification report 7ca3c1df P2.3.

    The LLM span is emitted by an external instrumentor (litellm/anthropic)
    which doesn't know about ``gen_ai.framework=cua``. The CUA callback
    attaches the framework to OTel baggage in ``on_run_start``; a span
    processor registered by ``CuaInstrumentor`` copies it onto any span
    missing the attribute. Simulate an external LLM span started while the
    CUA callback's baggage is active and verify it picks up ``cua``.
    """
    from opentelemetry.instrumentation.cua import (
        CuaInstrumentor,
        _register_framework_span_processor,
    )
    from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
        GEN_AI_SPAN_KIND,
    )

    _register_framework_span_processor(tracer_provider)
    CuaInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        agent = StubAgent(instructions="hi")
        cb = ArmsCuaCallback(handler, agent)

        async def scenario():
            await cb.on_run_start(
                {"model": "anthropic/claude-sonnet-4-5", "messages": "hello"}, []
            )
            # Simulate an external instrumentor (litellm) creating an LLM
            # span while the baggage context attached by on_run_start is
            # active. The span has no gen_ai.framework attribute of its own.
            tracer = tracer_provider.get_tracer("external-litellm")
            span = tracer.start_span(
                "chat claude-sonnet-4-5",
                attributes={GEN_AI_SPAN_KIND: "LLM"},
            )
            span.end()
            await cb.on_run_end({}, [], [])

        _run(scenario())

        spans = span_exporter.get_finished_spans()
        llm = next(s for s in spans if s.name == "chat claude-sonnet-4-5")
        attrs = dict(llm.attributes or {})
        assert attrs.get("gen_ai.framework") == "cua", (
            "LLM span should inherit gen_ai.framework=cua from baggage"
        )
    finally:
        CuaInstrumentor().uninstrument()

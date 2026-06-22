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

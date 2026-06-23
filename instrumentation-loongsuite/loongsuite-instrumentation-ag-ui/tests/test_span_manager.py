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

"""Tests for :class:`AGUISpanManager` event state machine."""

from __future__ import annotations

from types import SimpleNamespace

from opentelemetry.instrumentation.ag_ui.config import AGUIConfig
from opentelemetry.instrumentation.ag_ui.internal._span_manager import (
    AGUISpanManager,
)


def _evt(type_: str, **fields):
    return SimpleNamespace(type=type_, **fields)


def _input(**fields):
    defaults = {
        "thread_id": "t1",
        "run_id": "r1",
        "messages": [],
        "tools": [],
        "forwarded_props": {},
    }
    defaults.update(fields)
    return SimpleNamespace(**defaults)


def test_run_started_then_finished_yields_entry_and_agent(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t1", run_id="r1"))
    manager.on_event(_evt("RUN_FINISHED", thread_id="t1", run_id="r1"))

    spans = span_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "enter_ai_application_system" in names
    assert any("invoke_agent" in name for name in names)


def test_step_started_and_finished_creates_step_span(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(_evt("STEP_STARTED", step_name="plan"))
    manager.on_event(_evt("STEP_FINISHED", step_name="plan"))
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))

    spans = span_exporter.get_finished_spans()
    step_spans = [s for s in spans if s.name == "react step"]
    assert len(step_spans) == 1
    assert step_spans[0].attributes["gen_ai.react.round"] == 1
    assert step_spans[0].attributes["ag_ui.step.name"] == "plan"


def test_step_counter_increments(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(_evt("STEP_STARTED", step_name="s1"))
    manager.on_event(_evt("STEP_FINISHED", step_name="s1"))
    manager.on_event(_evt("STEP_STARTED", step_name="s2"))
    manager.on_event(_evt("STEP_FINISHED", step_name="s2"))
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))

    spans = span_exporter.get_finished_spans()
    rounds = sorted(
        s.attributes["gen_ai.react.round"]
        for s in spans
        if s.name == "react step"
    )
    assert rounds == [1, 2]


def test_tool_call_lifecycle(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=True),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(
        _evt(
            "TOOL_CALL_START",
            tool_call_id="tc1",
            tool_call_name="get_weather",
            parent_message_id="m1",
        )
    )
    manager.on_event(
        _evt("TOOL_CALL_ARGS", tool_call_id="tc1", delta='{"city":"Paris"}')
    )
    manager.on_event(_evt("TOOL_CALL_END", tool_call_id="tc1"))
    manager.on_event(
        _evt("TOOL_CALL_RESULT", tool_call_id="tc1", content="rainy")
    )
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))

    spans = span_exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name == "execute_tool get_weather"]
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["gen_ai.tool.name"] == "get_weather"
    assert tool_spans[0].attributes["gen_ai.tool.call.id"] == "tc1"
    assert tool_spans[0].attributes["gen_ai.tool.type"] == "function"
    assert "Paris" in tool_spans[0].attributes["gen_ai.tool.call.arguments"]
    assert "rainy" in tool_spans[0].attributes["gen_ai.tool.call.result"]


def test_tool_call_result_without_end_closes_tool(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(
        _evt(
            "TOOL_CALL_START",
            tool_call_id="tc1",
            tool_call_name="search",
        )
    )
    manager.on_event(
        _evt("TOOL_CALL_RESULT", tool_call_id="tc1", content="ok")
    )
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))

    spans = span_exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name == "execute_tool search"]
    assert len(tool_spans) == 1


def test_run_error_fails_open_spans(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(_evt("STEP_STARTED", step_name="plan"))
    manager.on_event(
        _evt(
            "TOOL_CALL_START",
            tool_call_id="tc1",
            tool_call_name="search",
        )
    )
    manager.on_event(_evt("RUN_ERROR", message="boom", code="E1"))

    spans = span_exporter.get_finished_spans()
    for span in spans:
        # All finished spans must end in error when RUN_ERROR arrives.
        if span.name in {"react step", "execute_tool search"}:
            assert span.status.is_ok is False


def test_text_message_content_records_ttft(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(
        _evt("TEXT_MESSAGE_CONTENT", message_id="m1", delta="hello")
    )
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))

    spans = span_exporter.get_finished_spans()
    entry = next(s for s in spans if s.name == "enter_ai_application_system")
    assert "gen_ai.response.time_to_first_token" in entry.attributes
    assert entry.attributes["gen_ai.response.time_to_first_token"] > 0


def test_unknown_event_is_ignored(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(_evt("STATE_SNAPSHOT", snapshot={}))
    manager.on_event(_evt("CUSTOM", name="x", value={}))
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))
    spans = span_exporter.get_finished_spans()
    assert {s.name for s in spans} == {
        "enter_ai_application_system",
        "invoke_agent ag-ui",
    }


def test_cleanup_closes_open_spans(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(_evt("STEP_STARTED", step_name="plan"))
    manager.cleanup()

    spans = span_exporter.get_finished_spans()
    assert any(s.name == "react step" for s in spans)
    assert any(s.name == "enter_ai_application_system" for s in spans)


def test_capture_content_records_input_messages(handler, span_exporter):
    input_data = _input(
        messages=[
            SimpleNamespace(role="user", content="hi"),
        ],
        tools=[
            SimpleNamespace(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object"},
            )
        ],
    )
    manager = AGUISpanManager(
        handler=handler,
        input_data=input_data,
        config=AGUIConfig(capture_content=True),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))

    spans = span_exporter.get_finished_spans()
    entry = next(s for s in spans if s.name == "enter_ai_application_system")
    assert "gen_ai.input.messages" in entry.attributes
    assert "hi" in entry.attributes["gen_ai.input.messages"]
    agent = next(s for s in spans if "invoke_agent" in s.name)
    assert agent.attributes["gen_ai.framework"] == "ag-ui"


def test_session_id_and_user_id_from_input(handler, span_exporter):
    input_data = _input(forwarded_props={"userId": "u-1"})
    manager = AGUISpanManager(
        handler=handler,
        input_data=input_data,
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="thread-42", run_id="r-1"))
    manager.on_event(_evt("RUN_FINISHED", thread_id="thread-42", run_id="r-1"))

    spans = span_exporter.get_finished_spans()
    entry = next(s for s in spans if s.name == "enter_ai_application_system")
    assert entry.attributes["gen_ai.session.id"] == "thread-42"
    assert entry.attributes["gen_ai.user.id"] == "u-1"
    agent = next(s for s in spans if "invoke_agent" in s.name)
    assert agent.attributes["gen_ai.agent.id"] == "r-1"
    assert agent.attributes["gen_ai.conversation.id"] == "thread-42"


# --- P2-1: TTFT must fire on TEXT_MESSAGE_CHUNK, not just CONTENT ----------
def test_text_message_chunk_records_ttft(handler, span_exporter):
    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(
        _evt("TEXT_MESSAGE_CHUNK", message_id="m1", delta="hello")
    )
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))

    spans = span_exporter.get_finished_spans()
    entry = next(s for s in spans if s.name == "enter_ai_application_system")
    assert "gen_ai.response.time_to_first_token" in entry.attributes
    assert entry.attributes["gen_ai.response.time_to_first_token"] > 0


# --- P2-2: gen_ai.tool.definitions must be recorded even when capture_content is off ---
def test_tool_definitions_recorded_without_capture_content(handler, span_exporter):
    input_data = _input(
        tools=[
            SimpleNamespace(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object"},
            )
        ],
    )
    manager = AGUISpanManager(
        handler=handler,
        input_data=input_data,
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))

    spans = span_exporter.get_finished_spans()
    agent = next(s for s in spans if "invoke_agent" in s.name)
    assert "gen_ai.tool.definitions" in agent.attributes
    assert "get_weather" in agent.attributes["gen_ai.tool.definitions"]


# --- P2-3: success path must set span status to OK -------------------------
def test_success_path_sets_ok_status(handler, span_exporter):
    from opentelemetry.trace.status import StatusCode

    manager = AGUISpanManager(
        handler=handler,
        input_data=_input(),
        config=AGUIConfig(capture_content=False),
    )
    manager.on_event(_evt("RUN_STARTED", thread_id="t", run_id="r"))
    manager.on_event(_evt("STEP_STARTED", step_name="plan"))
    manager.on_event(_evt("STEP_FINISHED", step_name="plan"))
    manager.on_event(_evt("RUN_FINISHED", thread_id="t", run_id="r"))

    spans = span_exporter.get_finished_spans()
    for span in spans:
        if span.name in {
            "enter_ai_application_system",
            "invoke_agent ag-ui",
            "react step",
        }:
            assert span.status.status_code is StatusCode.OK, (
                f"span {span.name} status was {span.status.status_code}"
            )


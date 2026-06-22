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

"""Tests for the AG-UI extractor helpers."""

from __future__ import annotations

from types import SimpleNamespace

from opentelemetry.instrumentation.ag_ui.internal._extractors import (
    convert_agui_messages_to_input_messages,
    convert_agui_messages_to_output_messages,
    convert_agui_tools_to_definitions,
    forwarded_user_id,
    serialize_content,
)


def _msg(**fields):
    return SimpleNamespace(**fields)


def _input_content(text):
    return SimpleNamespace(type="text", text=text)


def test_convert_user_message_text():
    messages = [
        _msg(role="user", content="hello"),
        _msg(role="assistant", content="hi", tool_calls=None),
    ]
    result = convert_agui_messages_to_input_messages(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].parts[0].content == "hello"
    assert result[1].role == "assistant"
    assert result[1].parts[0].content == "hi"


def test_convert_user_message_multimodal():
    messages = [
        _msg(
            role="user",
            content=[_input_content("a"), _input_content("b")],
        ),
    ]
    result = convert_agui_messages_to_input_messages(messages)
    assert len(result) == 1
    assert [p.content for p in result[0].parts] == ["a", "b"]


def test_convert_assistant_message_with_tool_calls():
    function = SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}')
    tool_call = SimpleNamespace(id="tc1", function=function)
    messages = [
        _msg(role="assistant", content=None, tool_calls=[tool_call]),
    ]
    result = convert_agui_messages_to_input_messages(messages)
    assert result[0].role == "assistant"
    assert result[0].parts[0].name == "get_weather"
    assert result[0].parts[0].id == "tc1"
    assert result[0].parts[0].arguments == '{"city":"Paris"}'


def test_convert_tool_message_to_tool_call_response():
    messages = [
        _msg(role="tool", content="rainy", tool_call_id="tc1"),
    ]
    result = convert_agui_messages_to_input_messages(messages)
    assert result[0].role == "tool"
    assert result[0].parts[0].response == "rainy"
    assert result[0].parts[0].id == "tc1"


def test_activity_message_skipped():
    messages = [
        _msg(role="activity", content={}, activity_type="thinking"),
        _msg(role="user", content="hi"),
    ]
    result = convert_agui_messages_to_input_messages(messages)
    assert len(result) == 1
    assert result[0].role == "user"


def test_empty_messages_returns_empty_list():
    assert convert_agui_messages_to_input_messages(None) == []
    assert convert_agui_messages_to_input_messages([]) == []


def test_output_messages_finish_reasons():
    messages = [_msg(role="assistant", content="done", tool_calls=None)]
    result = convert_agui_messages_to_output_messages(messages, run_error=False)
    assert result[0].finish_reason == "stop"

    function = SimpleNamespace(name="x", arguments="{}")
    tc = SimpleNamespace(id="t1", function=function)
    messages = [_msg(role="assistant", content=None, tool_calls=[tc])]
    result = convert_agui_messages_to_output_messages(messages, run_error=False)
    assert result[0].finish_reason == "tool_calls"

    result = convert_agui_messages_to_output_messages(
        [_msg(role="assistant", content="x", tool_calls=None)],
        run_error=True,
    )
    assert result[0].finish_reason == "error"


def test_convert_tools_with_capture():
    tools = [
        SimpleNamespace(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object"},
        ),
        SimpleNamespace(
            name="search",
            description="Search",
            parameters=None,
        ),
    ]
    result = convert_agui_tools_to_definitions(tools, capture_content=True)
    assert len(result) == 2
    assert result[0].description == "Get weather"
    assert result[0].parameters == {"type": "object"}
    assert result[0].type == "function"


def test_convert_tools_without_capture_omits_description():
    tools = [
        SimpleNamespace(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object"},
        ),
    ]
    result = convert_agui_tools_to_definitions(tools, capture_content=False)
    assert len(result) == 1
    assert result[0].description is None
    assert result[0].parameters is None
    assert result[0].name == "get_weather"


def test_forwarded_user_id_from_dict():
    input_data = SimpleNamespace(forwarded_props={"userId": "u-123"})
    assert forwarded_user_id(input_data) == "u-123"


def test_forwarded_user_id_from_object():
    forwarded = SimpleNamespace(userId="u-abc")
    input_data = SimpleNamespace(forwarded_props=forwarded)
    assert forwarded_user_id(input_data) == "u-abc"


def test_forwarded_user_id_missing():
    assert forwarded_user_id(None) is None
    assert forwarded_user_id(SimpleNamespace(forwarded_props=None)) is None
    assert (
        forwarded_user_id(SimpleNamespace(forwarded_props={"foo": "bar"}))
        is None
    )


def test_serialize_content_truncates_long_strings():
    text = "x" * 100
    serialized = serialize_content(text, max_length=10)
    assert serialized is not None
    assert serialized.endswith("...<truncated>")
    assert len(serialized) < len(text)


def test_serialize_content_returns_none_for_none():
    assert serialize_content(None, max_length=10) is None


def test_serialize_content_handles_complex_objects():
    serialized = serialize_content({"key": "value"}, max_length=1000)
    assert serialized == '{"key": "value"}'

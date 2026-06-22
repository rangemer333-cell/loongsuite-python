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

"""Extractors that convert AG-UI protocol types into util-genai invocation data.

These helpers are intentionally defensive: AG-UI events are produced by many
upstream integrations (LangGraph, CrewAI, Strands, Claude SDK, ...) and field
availability is not guaranteed. Every accessor returns ``None`` rather than
raising so that the :class:`AGUISpanManager` can omit attributes that cannot
be collected.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _convert_input_content(content: Any) -> list[MessagePart]:
    parts: list[MessagePart] = []
    if content is None:
        return parts
    if _is_str(content):
        if content:
            parts.append(Text(content=content))
        return parts
    if isinstance(content, list):
        for item in content:
            if item is None:
                continue
            # TextInputContent-like fragment with `type="text"` and `text`
            item_type = _safe_get(item, "type")
            if item_type == "text":
                text_value = _safe_get(item, "text")
                if _is_str(text_value) and text_value:
                    parts.append(Text(content=text_value))
                continue
            # ImageInputContent / AudioInputContent / ... — fall back to a
            # textual placeholder so that the message role is still recorded.
            label = item_type or "multimodal"
            parts.append(Text(content=f"<{label}>"))
    return parts


def _convert_agui_message_to_input_message(msg: Any) -> InputMessage | None:
    if msg is None:
        return None
    role = _safe_get(msg, "role")
    if not role:
        return None
    parts: list[MessagePart] = []

    if role == "user":
        parts.extend(_convert_input_content(_safe_get(msg, "content")))
    elif role == "assistant":
        content = _safe_get(msg, "content")
        if _is_str(content) and content:
            parts.append(Text(content=content))
        tool_calls = _safe_get(msg, "tool_calls") or []
        for tc in tool_calls:
            tc_id = _safe_get(tc, "id")
            function = _safe_get(tc, "function")
            if function is None:
                continue
            name = _safe_get(function, "name")
            arguments = _safe_get(function, "arguments")
            parts.append(
                ToolCall(id=tc_id, name=name or "", arguments=arguments)
            )
    elif role == "tool":
        tool_call_id = _safe_get(msg, "tool_call_id")
        content = _safe_get(msg, "content")
        parts.append(ToolCallResponse(id=tool_call_id, response=content))
    elif role in ("system", "developer"):
        content = _safe_get(msg, "content")
        if _is_str(content) and content:
            parts.append(Text(content=content))
    elif role == "reasoning":
        content = _safe_get(msg, "content")
        if _is_str(content) and content:
            parts.append(Text(content=content))
    elif role == "activity":
        # Activity messages are not part of the input.messages schema.
        return None

    return InputMessage(role=role, parts=parts)


def convert_agui_messages_to_input_messages(
    messages: Iterable[Any] | None,
) -> list[InputMessage]:
    if not messages:
        return []
    result: list[InputMessage] = []
    for msg in messages:
        converted = _convert_agui_message_to_input_message(msg)
        if converted is not None:
            result.append(converted)
    return result


def _finish_reason_from_message(msg: Any, run_error: bool) -> str:
    if run_error:
        return "error"
    role = _safe_get(msg, "role")
    tool_calls = _safe_get(msg, "tool_calls")
    if role == "assistant" and tool_calls:
        return "tool_calls"
    return "stop"


def convert_agui_messages_to_output_messages(
    messages: Iterable[Any] | None, run_error: bool = False
) -> list[OutputMessage]:
    if not messages:
        return []
    result: list[OutputMessage] = []
    for msg in messages:
        if msg is None:
            continue
        role = _safe_get(msg, "role")
        if role is None:
            continue
        parts: list[MessagePart] = []
        if role == "user":
            parts.extend(_convert_input_content(_safe_get(msg, "content")))
        elif role == "assistant":
            content = _safe_get(msg, "content")
            if _is_str(content) and content:
                parts.append(Text(content=content))
            for tc in _safe_get(msg, "tool_calls") or []:
                function = _safe_get(tc, "function")
                if function is None:
                    continue
                parts.append(
                    ToolCall(
                        id=_safe_get(tc, "id"),
                        name=_safe_get(function, "name") or "",
                        arguments=_safe_get(function, "arguments"),
                    )
                )
        elif role == "tool":
            parts.append(
                ToolCallResponse(
                    id=_safe_get(msg, "tool_call_id"),
                    response=_safe_get(msg, "content"),
                )
            )
        elif role in ("system", "developer", "reasoning"):
            content = _safe_get(msg, "content")
            if _is_str(content) and content:
                parts.append(Text(content=content))
        elif role == "activity":
            continue

        if not parts:
            continue
        result.append(
            OutputMessage(
                role=role,
                parts=parts,
                finish_reason=_finish_reason_from_message(msg, run_error),
            )
        )
    return result


def convert_agui_tools_to_definitions(
    tools: Iterable[Any] | None, capture_content: bool
) -> list[FunctionToolDefinition]:
    if not tools:
        return []
    definitions: list[FunctionToolDefinition] = []
    for tool in tools:
        if tool is None:
            continue
        name = _safe_get(tool, "name")
        if not name:
            continue
        description = _safe_get(tool, "description") if capture_content else None
        parameters = _safe_get(tool, "parameters") if capture_content else None
        definitions.append(
            FunctionToolDefinition(
                name=name, description=description, parameters=parameters
            )
        )
    return definitions


def _truncate(value: str, max_length: int) -> tuple[str, bool]:
    if not isinstance(value, str):
        return value, False
    if len(value) <= max_length:
        return value, False
    return value[:max_length], True


def serialize_content(value: Any, max_length: int) -> str | None:
    """Serialize a structured value into a JSON string for span attributes.

    Returns ``None`` when the value cannot be serialized. Truncates the
    resulting string to ``max_length`` characters and appends a marker when
    truncation occurs.
    """
    if value is None:
        return None
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None
    text, truncated = _truncate(text, max_length)
    if truncated:
        text = text + "...<truncated>"
    return text


def forwarded_user_id(input_data: Any) -> str | None:
    """Extract ``gen_ai.user.id`` from ``RunAgentInput.forwarded_props``."""
    if input_data is None:
        return None
    forwarded = _safe_get(input_data, "forwarded_props")
    if forwarded is None:
        return None
    if isinstance(forwarded, dict):
        user_id = forwarded.get("userId") or forwarded.get("user_id")
        if _is_str(user_id) and user_id:
            return user_id
        return None
    user_id = _safe_get(forwarded, "userId") or _safe_get(forwarded, "user_id")
    if _is_str(user_id) and user_id:
        return user_id
    return None

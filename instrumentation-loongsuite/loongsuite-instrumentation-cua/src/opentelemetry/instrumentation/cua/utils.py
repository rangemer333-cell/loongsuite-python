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

"""Helpers for CUA instrumentation: attribute extraction and conversion."""

import hashlib
import json
from typing import Any, Dict, Optional

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import get_content_capturing_mode


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def extract_provider_from_model(model: str) -> str:
    """Infer provider from CUA model string (e.g. 'anthropic/claude-...' -> 'anthropic')."""
    if not model:
        return "unknown"
    if "/" in model:
        provider = model.split("/", 1)[0]
        return provider or "unknown"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    if model.startswith("gemini"):
        return "google"
    return "unknown"


def compute_agent_id(model: str, instructions: Optional[str]) -> Optional[str]:
    """Stable agent id derived from model + instructions."""
    if not model:
        return None
    raw = f"{model}|{instructions or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_ACTION_DESCRIPTIONS = {
    "click": "Click at coordinates",
    "double_click": "Double click at coordinates",
    "triple_click": "Triple click at coordinates",
    "right_click": "Right click at coordinates",
    "middle_click": "Middle click at coordinates",
    "type": "Type text",
    "keypress": "Press keys",
    "key": "Press keys",
    "scroll": "Scroll",
    "wait": "Wait",
    "screenshot": "Take screenshot",
    "cursor_position": "Get cursor position",
    "mouse_move": "Move mouse",
    "hold_key": "Hold key",
    "hold_mouse": "Hold mouse button",
    "release_mouse": "Release mouse button",
}


def action_description(action_type: str) -> str:
    return _ACTION_DESCRIPTIONS.get(action_type, "Computer action")


def should_capture_content() -> bool:
    """Return True when content capture is enabled (SPAN_ONLY or SPAN_AND_EVENT)."""
    try:
        mode = get_content_capturing_mode()
    except Exception:
        mode = None
    return mode in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    )


def _set_optional_action_attrs(inv_attributes: Dict[str, Any], action: Dict[str, Any]) -> None:
    """Populate cua.action.* extension attributes on a tool invocation attributes dict."""
    if not isinstance(action, dict):
        return
    action_type = action.get("type")
    if action_type is not None:
        inv_attributes["cua.action.type"] = action_type

    x = action.get("x")
    y = action.get("y")
    if x is not None or y is not None:
        inv_attributes["cua.action.coordinate"] = f"({x},{y})"

    if "button" in action:
        inv_attributes["cua.action.button"] = action.get("button", "left")
    if "text" in action and should_capture_content():
        inv_attributes["cua.action.text"] = action.get("text")
    if "keys" in action:
        keys = action.get("keys")
        try:
            inv_attributes["cua.action.keys"] = (
                keys if isinstance(keys, str) else json.dumps(keys, default=str)
            )
        except (TypeError, ValueError):
            inv_attributes["cua.action.keys"] = str(keys)


def serialize_arguments(value: Any) -> Optional[str]:
    """Serialize tool arguments/results for the standard gen_ai attributes.

    Returns None when content capture is disabled or serialization fails.
    """
    if not should_capture_content():
        return None
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def extract_finish_reason(responses: Any) -> str:
    """Extract a finish_reason from the CUA responses dict.

    Heuristics: look at the last item in responses['output']:
    - if role == assistant and has tool_calls -> 'tool_calls'
    - if role == assistant and no tool_calls -> 'stop'
    - otherwise -> 'unknown'
    """
    if not isinstance(responses, dict):
        return "unknown"
    output = responses.get("output") or responses.get("items")
    if not isinstance(output, list) or not output:
        return "unknown"
    last = output[-1]
    if isinstance(last, dict):
        role = last.get("role") or last.get("type")
        has_tool_calls = bool(last.get("tool_calls") or last.get("tool_use"))
        if role == "assistant":
            return "tool_calls" if has_tool_calls else "stop"
        return "unknown"
    return "unknown"


def env_content_capture_var() -> str:
    return OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT

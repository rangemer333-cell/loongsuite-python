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

"""Stateful mapper from AG-UI event stream to OpenTelemetry Span lifecycle.

The :class:`AGUISpanManager` is owned by a single ``EventEncoder`` instance
(per-request) and consumes the typed events flowing through ``encode()``.
It builds four levels of spans — ENTRY → AGENT → STEP → TOOL — using the
LoongSuite :class:`ExtendedTelemetryHandler` Pattern A API.

Span lifecycle is driven by the AG-UI event type:

    RUN_STARTED      → open ENTRY + AGENT
    STEP_STARTED      → open STEP
    TOOL_CALL_START   → open TOOL
    TOOL_CALL_ARGS    → accumulate tool arguments
    TOOL_CALL_END / TOOL_CALL_RESULT → close TOOL
    STEP_FINISHED     → close STEP
    TEXT_MESSAGE_CONTENT → first-token timing (no span)
    MESSAGES_SNAPSHOT → cache last snapshot for output.messages
    RUN_FINISHED     → close AGENT + ENTRY
    RUN_ERROR        → fail AGENT + ENTRY
"""

from __future__ import annotations

import json
import logging
import timeit
from typing import Any, Callable

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.types import Error

from opentelemetry.instrumentation.ag_ui.config import AGUIConfig
from opentelemetry.instrumentation.ag_ui.internal._extractors import (
    convert_agui_messages_to_input_messages,
    convert_agui_messages_to_output_messages,
    convert_agui_tools_to_definitions,
    forwarded_user_id,
    serialize_content,
)

logger = logging.getLogger(__name__)

AG_UI_FRAMEWORK = "ag-ui"
AG_UI_AGENT_NAME = "ag-ui"


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _try_parse_json(text: str) -> Any:
    if not text:
        return text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


class _ToolState:
    """Per-``tool_call_id`` accumulator for arguments and result."""

    __slots__ = ("invocation", "args_chunks", "result", "ended", "end_received")

    def __init__(self, invocation: ExecuteToolInvocation) -> None:
        self.invocation = invocation
        self.args_chunks: list[str] = []
        self.result: Any = None
        self.ended: bool = False
        self.end_received: bool = False

    def append_args(self, delta: str, max_length: int) -> None:
        if not isinstance(delta, str) or not delta:
            return
        current_len = sum(len(chunk) for chunk in self.args_chunks)
        if current_len >= max_length:
            return
        remaining = max_length - current_len
        if len(delta) > remaining:
            delta = delta[:remaining]
        self.args_chunks.append(delta)


class AGUISpanManager:
    """AG-UI event flow → OTel Span lifecycle state machine."""

    def __init__(
        self,
        handler: ExtendedTelemetryHandler,
        input_data: Any,
        config: AGUIConfig,
    ) -> None:
        self._handler = handler
        self._input_data = input_data
        self._config = config

        self._entry_inv: EntryInvocation | None = None
        self._agent_inv: InvokeAgentInvocation | None = None
        self._step_invs: dict[str, ReactStepInvocation] = {}
        self._tool_states: dict[str, _ToolState] = {}

        self._last_messages_snapshot: Any = None
        self._step_counter: int = 0
        self._entry_start_time: float | None = None
        self._first_content_time: float | None = None
        self._closed: bool = False
        self._dispatch: dict[str, Callable[[Any], None]] = self._build_dispatch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_event(self, event: Any) -> None:
        if event is None or self._closed:
            return
        try:
            event_type = _safe_get(event, "type")
            if event_type is None:
                return
            method = self._dispatch.get(event_type)
            if method is None:
                return
            method(event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "AG-UI instrumentation failed to handle event: %s",
                exc,
                exc_info=True,
            )

    def cleanup(self) -> None:
        """Force-close any still-open spans (stream interrupted / encoder GC)."""
        if self._closed:
            return
        self._closed = True
        error = Error(message="AG-UI stream interrupted", type=RuntimeError)
        for tool_state in list(self._tool_states.values()):
            if not tool_state.ended:
                self._safe_fail_tool(tool_state.invocation, error)
        for step_inv in list(self._step_invs.values()):
            self._safe_call(
                "fail_react_step",
                self._handler.fail_react_step,
                step_inv,
                error,
            )
        if self._agent_inv is not None:
            self._safe_call(
                "fail_invoke_agent",
                self._handler.fail_invoke_agent,
                self._agent_inv,
                error,
            )
            self._agent_inv = None
        if self._entry_inv is not None:
            self._safe_call(
                "fail_entry", self._handler.fail_entry, self._entry_inv, error
            )
            self._entry_inv = None
        self._step_invs.clear()
        self._tool_states.clear()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_run_started(self, event: Any) -> None:
        if self._entry_inv is not None:
            return
        thread_id = _safe_get(event, "thread_id")
        run_id = _safe_get(event, "run_id")
        user_id = forwarded_user_id(self._input_data)

        entry_inv = EntryInvocation(
            session_id=thread_id,
            user_id=user_id,
        )
        entry_inv.attributes["gen_ai.framework"] = AG_UI_FRAMEWORK

        if self._config.capture_content and self._input_data is not None:
            input_messages = convert_agui_messages_to_input_messages(
                _safe_get(self._input_data, "messages")
            )
            serialized = serialize_content(
                [m.__dict__ for m in input_messages],
                self._config.max_content_length,
            )
            if serialized is not None:
                entry_inv.attributes["gen_ai.input.messages"] = serialized

        self._safe_call("start_entry", self._handler.start_entry, entry_inv)
        self._entry_inv = entry_inv
        self._entry_start_time = timeit.default_timer()

        agent_inv = InvokeAgentInvocation(provider=AG_UI_FRAMEWORK)
        agent_inv.agent_name = AG_UI_AGENT_NAME
        agent_inv.agent_id = run_id
        agent_inv.conversation_id = thread_id
        agent_inv.attributes["gen_ai.framework"] = AG_UI_FRAMEWORK
        if self._config.capture_content and self._input_data is not None:
            input_messages = convert_agui_messages_to_input_messages(
                _safe_get(self._input_data, "messages")
            )
            agent_inv.input_messages = input_messages
            tool_defs = convert_agui_tools_to_definitions(
                _safe_get(self._input_data, "tools"),
                capture_content=self._config.capture_content,
            )
            agent_inv.tool_definitions = tool_defs
        self._safe_call(
            "start_invoke_agent", self._handler.start_invoke_agent, agent_inv
        )
        self._agent_inv = agent_inv

    def _on_run_finished(self, event: Any) -> None:
        if self._entry_inv is None and self._agent_inv is None:
            return
        self._finalize_agent_and_entry(run_error=False)

    def _on_run_error(self, event: Any) -> None:
        message = _safe_get(event, "message") or "AG-UI run error"
        code = _safe_get(event, "code")
        error = Error(message=message, type=RuntimeError)

        for step_inv in list(self._step_invs.values()):
            step_inv.finish_reason = "error"

        for tool_state in list(self._tool_states.values()):
            if not tool_state.ended:
                self._safe_fail_tool(tool_state.invocation, error)

        for step_inv in list(self._step_invs.values()):
            self._safe_call(
                "fail_react_step",
                self._handler.fail_react_step,
                step_inv,
                error,
            )
        self._step_invs.clear()
        self._tool_states.clear()

        if self._agent_inv is not None:
            if self._agent_inv.span is not None and code:
                self._agent_inv.attributes["error.code"] = code
            self._safe_call(
                "fail_invoke_agent",
                self._handler.fail_invoke_agent,
                self._agent_inv,
                error,
            )
            self._agent_inv = None
        if self._entry_inv is not None:
            self._safe_call(
                "fail_entry", self._handler.fail_entry, self._entry_inv, error
            )
            self._entry_inv = None

    def _on_step_started(self, event: Any) -> None:
        step_name = _safe_get(event, "step_name")
        if not step_name:
            return
        if step_name in self._step_invs:
            return
        self._step_counter += 1
        step_inv = ReactStepInvocation(round=self._step_counter)
        step_inv.attributes["gen_ai.framework"] = AG_UI_FRAMEWORK
        step_inv.attributes["ag_ui.step.name"] = step_name
        self._safe_call(
            "start_react_step", self._handler.start_react_step, step_inv
        )
        self._step_invs[step_name] = step_inv

    def _on_step_finished(self, event: Any) -> None:
        step_name = _safe_get(event, "step_name")
        if not step_name:
            return
        step_inv = self._step_invs.pop(step_name, None)
        if step_inv is None:
            return
        self._safe_call(
            "stop_react_step", self._handler.stop_react_step, step_inv
        )

    def _on_tool_call_start(self, event: Any) -> None:
        tool_call_id = _safe_get(event, "tool_call_id")
        tool_name = _safe_get(event, "tool_call_name")
        if not tool_call_id:
            return
        if tool_call_id in self._tool_states:
            return
        tool_inv = ExecuteToolInvocation(
            tool_name=tool_name or "unknown",
            tool_call_id=tool_call_id,
            tool_type="function",
        )
        tool_inv.attributes["gen_ai.framework"] = AG_UI_FRAMEWORK
        self._safe_call(
            "start_execute_tool", self._handler.start_execute_tool, tool_inv
        )
        self._tool_states[tool_call_id] = _ToolState(tool_inv)

    def _on_tool_call_args(self, event: Any) -> None:
        tool_call_id = _safe_get(event, "tool_call_id")
        delta = _safe_get(event, "delta")
        state = self._tool_states.get(tool_call_id) if tool_call_id else None
        if state is None:
            return
        state.append_args(delta, self._config.max_content_length)

    def _on_tool_call_end(self, event: Any) -> None:
        tool_call_id = _safe_get(event, "tool_call_id")
        if not tool_call_id:
            return
        state = self._tool_states.get(tool_call_id)
        if state is None or state.ended:
            return
        # Per AG-UI spec: the TOOL span closes on TOOL_CALL_RESULT because
        # that event carries the tool's execution result. If END arrives
        # first, mark and wait; the span will close on the subsequent RESULT
        # event (or be force-closed by RUN_FINISHED / cleanup if no result
        # ever arrives, e.g. for frontend-only tool calls).
        state.end_received = True
        if state.result is not None:
            self._close_tool(state, error=None)

    def _on_tool_call_result(self, event: Any) -> None:
        tool_call_id = _safe_get(event, "tool_call_id")
        content = _safe_get(event, "content")
        state = self._tool_states.get(tool_call_id) if tool_call_id else None
        if state is None:
            return
        state.result = content
        self._close_tool(state, error=None)

    def _on_text_content(self, event: Any) -> None:
        if self._first_content_time is not None:
            return
        self._first_content_time = timeit.default_timer()
        if (
            self._entry_inv is not None
            and self._entry_start_time is not None
            and self._entry_inv.span is not None
        ):
            ttft_ns = int(
                (self._first_content_time - self._entry_start_time) * 1e9
            )
            self._entry_inv.response_time_to_first_token = ttft_ns
            if self._agent_inv is not None:
                self._agent_inv.monotonic_first_token_s = (
                    self._first_content_time
                )

    def _on_messages_snapshot(self, event: Any) -> None:
        self._last_messages_snapshot = event

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_dispatch(self) -> dict[str, Callable[[Any], None]]:
        return {
            "RUN_STARTED": self._on_run_started,
            "RUN_FINISHED": self._on_run_finished,
            "RUN_ERROR": self._on_run_error,
            "STEP_STARTED": self._on_step_started,
            "STEP_FINISHED": self._on_step_finished,
            "TOOL_CALL_START": self._on_tool_call_start,
            "TOOL_CALL_ARGS": self._on_tool_call_args,
            "TOOL_CALL_END": self._on_tool_call_end,
            "TOOL_CALL_RESULT": self._on_tool_call_result,
            "TEXT_MESSAGE_CONTENT": self._on_text_content,
            "MESSAGES_SNAPSHOT": self._on_messages_snapshot,
        }

    def _close_tool(self, state: _ToolState, error: Error | None) -> None:
        if state.ended:
            return
        state.ended = True
        inv = state.invocation
        if self._config.capture_content:
            args_text = "".join(state.args_chunks) if state.args_chunks else ""
            if args_text:
                inv.tool_call_arguments = serialize_content(
                    _try_parse_json(args_text),
                    self._config.max_content_length,
                ) or args_text
            if state.result is not None:
                inv.tool_call_result = serialize_content(
                    state.result, self._config.max_content_length
                )
        if error is not None:
            self._safe_fail_tool(inv, error)
            self._tool_states.pop(_safe_get(inv, "tool_call_id"), None)
            return
        self._safe_call(
            "stop_execute_tool", self._handler.stop_execute_tool, inv
        )
        self._tool_states.pop(_safe_get(inv, "tool_call_id"), None)

    def _safe_fail_tool(
        self, invocation: ExecuteToolInvocation, error: Error
    ) -> None:
        self._safe_call(
            "fail_execute_tool",
            self._handler.fail_execute_tool,
            invocation,
            error,
        )

    def _finalize_agent_and_entry(self, run_error: bool) -> None:
        for state in list(self._tool_states.values()):
            if not state.ended:
                self._close_tool(state, error=None)
        for step_name, step_inv in list(self._step_invs.items()):
            if run_error:
                step_inv.finish_reason = "error"
            self._safe_call(
                "stop_react_step", self._handler.stop_react_step, step_inv
            )
            self._step_invs.pop(step_name, None)

        if (
            self._config.capture_content
            and self._last_messages_snapshot is not None
            and self._agent_inv is not None
        ):
            output_messages = convert_agui_messages_to_output_messages(
                _safe_get(self._last_messages_snapshot, "messages"),
                run_error=run_error,
            )
            self._agent_inv.output_messages = output_messages
            if self._entry_inv is not None:
                self._entry_inv.output_messages = output_messages

        if self._agent_inv is not None:
            self._safe_call(
                "stop_invoke_agent",
                self._handler.stop_invoke_agent,
                self._agent_inv,
            )
            self._agent_inv = None
        if self._entry_inv is not None:
            self._safe_call(
                "stop_entry", self._handler.stop_entry, self._entry_inv
            )
            self._entry_inv = None

    def _safe_call(self, action: str, method: Any, *args: Any) -> bool:
        try:
            method(*args)
            return True
        except Exception as exc:
            logger.warning(
                "AG-UI instrumentation %s failed: %s", action, exc, exc_info=True
            )
            return False

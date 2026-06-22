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

"""ARMS GenAI semantic-conventions Callback for CUA ComputerAgent."""

import logging
import time
from uuid import uuid4
from typing import Any, Dict, List, Optional

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    EntryInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
)

from opentelemetry.instrumentation.cua.utils import (
    action_description,
    compute_agent_id,
    extract_finish_reason,
    extract_provider_from_model,
    _set_optional_action_attrs,
    serialize_arguments,
)

logger = logging.getLogger(__name__)

GEN_AI_FRAMEWORK = "gen_ai.framework"
_FRAMEWORK_VALUE = "cua"


def _text_part(text: Any) -> Text:
    return Text(content=str(text) if text is not None else "")


def _messages_to_input_messages(messages: Any) -> List[InputMessage]:
    """Convert a CUA/OpenAI-style messages list to ``InputMessage`` objects.

    CUA messages look like ``[{"role": "user", "content": "..."}]``. We accept
    both plain strings and lists of content parts (OpenAI format).
    """
    out: List[InputMessage] = []
    if not isinstance(messages, list):
        return out
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "unknown"
        content = msg.get("content")
        parts: List[Any] = []
        if content is None:
            continue
        if isinstance(content, str):
            parts.append(_text_part(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text is not None:
                        parts.append(_text_part(text))
                elif isinstance(part, str):
                    parts.append(_text_part(part))
        elif isinstance(content, dict):
            text = content.get("text") or content.get("content")
            if text is not None:
                parts.append(_text_part(text))
        if parts:
            out.append(InputMessage(role=role, parts=parts))
    return out


def _responses_to_output_messages(responses: Any) -> List[OutputMessage]:
    """Convert CUA ``responses`` dict to ``OutputMessage`` objects.

    ``responses`` typically looks like ``{"output": [{"role": "assistant",
    "content": "..."}]}``. We only emit a structured message when textual
    content is extractable; otherwise we return an empty list (the raw payload
    is still surfaced on the LLM child span produced by LiteLLM instrumentation).
    """
    out: List[OutputMessage] = []
    if not isinstance(responses, dict):
        return out
    items = responses.get("output") or responses.get("items")
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("type") or "assistant"
        content = item.get("content")
        parts: List[Any] = []
        if isinstance(content, str) and content:
            parts.append(_text_part(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text is not None:
                        parts.append(_text_part(text))
                elif isinstance(part, str):
                    parts.append(_text_part(part))
        if parts:
            finish_reason = item.get("finish_reason") or "stop"
            out.append(OutputMessage(role=role, parts=parts, finish_reason=finish_reason))
    return out


def _system_instruction_from_text(instructions: Any) -> List[Any]:
    if instructions is None:
        return []
    if isinstance(instructions, str):
        if not instructions:
            return []
        return [_text_part(instructions)]
    return []


def _extract_tool_definitions(agent: Any) -> List[Any]:
    """Convert CUA ``ComputerAgent.tool_schemas`` entries to OTel ToolDefinition.

    CUA's ``_process_tools`` emits entries shaped like
    ``{"type": "computer", "computer": <obj>}`` or
    ``{"type": "function", "function": {"name":..., "description":..., "parameters":...}}``.
    We map them to ``GenericToolDefinition`` / ``FunctionToolDefinition`` so
    the handler serializes ``gen_ai.tool.definitions`` (per gen-ai.md §[10/12]).
    """
    out: List[Any] = []
    schemas = getattr(agent, "tool_schemas", None)
    if not isinstance(schemas, list):
        return out
    for schema in schemas:
        if not isinstance(schema, dict):
            continue
        t = schema.get("type")
        if t == "function":
            fn = schema.get("function") or {}
            name = fn.get("name") or "function"
            description = fn.get("description")
            parameters = fn.get("parameters")
            out.append(
                FunctionToolDefinition(
                    name=name,
                    description=description,
                    parameters=parameters if parameters is not None else {},
                )
            )
        elif t:
            name = schema.get("name") or t
            out.append(GenericToolDefinition(name=name, type=t))
    return out


class ArmsCuaCallback:
    """AsyncCallbackHandler implementation that emits ARMS GenAI spans.

    Implements the union of AsyncCallbackHandler hooks used by ComputerAgent.
    We intentionally do NOT subclass the CUA base class to avoid an import
    dependency at instrument() time (the CUA package may not be installed
    when the instrumentor is enumerated by ``opentelemetry-instrument``).
    The duck-typed method set is sufficient — ComputerAgent iterates
    ``self.callbacks`` and awaits these coroutines by name.
    """

    def __init__(self, handler: ExtendedTelemetryHandler, agent: Any) -> None:
        self._handler = handler
        self._agent = agent
        self._entry_inv: Optional[EntryInvocation] = None
        self._agent_inv: Optional[InvokeAgentInvocation] = None
        self._step_inv: Optional[ReactStepInvocation] = None
        self._tool_inv: Optional[ExecuteToolInvocation] = None
        self._step_count = 0
        self._run_start_ns: int = 0
        self._first_token_ns: Optional[int] = None
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._input_messages: List[Dict[str, Any]] = []
        self._last_responses: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def on_run_start(self, kwargs: Dict[str, Any], old_items: List[Dict[str, Any]]) -> None:
        self._run_start_ns = time.perf_counter_ns()
        self._step_count = 0
        self._first_token_ns = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._input_messages = kwargs.get("messages") or []

        session_id = str(uuid4())
        self._entry_inv = EntryInvocation(
            session_id=session_id,
            user_id=kwargs.get("user_id"),
        )
        self._entry_inv.attributes[GEN_AI_FRAMEWORK] = _FRAMEWORK_VALUE
        try:
            self._handler.start_entry(self._entry_inv)
        except Exception as exc:
            logger.warning("CUA: start_entry failed: %s", exc)
            self._entry_inv = None
            return

        model = kwargs.get("model") or getattr(self._agent, "model", None) or "unknown"
        provider = extract_provider_from_model(model)
        instructions = getattr(self._agent, "instructions", None)
        agent_id = compute_agent_id(model, instructions)
        self._agent_inv = InvokeAgentInvocation(
            provider=provider,
            agent_name=model,
            agent_description=instructions,
            agent_id=agent_id,
            conversation_id=session_id,
            request_model=model,
        )
        self._agent_inv.attributes[GEN_AI_FRAMEWORK] = _FRAMEWORK_VALUE
        try:
            self._handler.start_invoke_agent(self._agent_inv)
        except Exception as exc:
            logger.warning("CUA: start_invoke_agent failed: %s", exc)
            self._agent_inv = None

    async def on_run_end(
        self,
        kwargs: Dict[str, Any],
        old_items: List[Dict[str, Any]],
        new_items: List[Dict[str, Any]],
    ) -> None:
        # Close any dangling step/tool spans
        self._fail_open_tool_span()
        self._close_open_step_span()

        # Convert captured input messages / responses into the structured
        # InputMessage / OutputMessage objects the handler knows how to
        # serialize as gen_ai.input.messages / gen_ai.output.messages.
        input_msgs = _messages_to_input_messages(self._input_messages)
        output_msgs = _responses_to_output_messages(self._last_responses)
        instructions = getattr(self._agent, "instructions", None)
        system_instruction = _system_instruction_from_text(instructions)
        tool_defs = _extract_tool_definitions(self._agent)

        if self._agent_inv is not None:
            self._agent_inv.input_tokens = self._input_tokens or None
            self._agent_inv.output_tokens = self._output_tokens or None
            # The handler computes TTFT as
            # ``(monotonic_first_token_s - monotonic_start_s) * 1e9`` where both
            # values are absolute monotonic times (seconds). Earlier we set
            # ``monotonic_first_token_s`` to a delta from ``_run_start_ns``,
            # which made the comparison fail and dropped the attribute on
            # AGENT. Use the absolute perf_counter reading instead.
            if self._first_token_ns is not None:
                self._agent_inv.monotonic_first_token_s = self._first_token_ns / 1e9
            if input_msgs:
                self._agent_inv.input_messages = input_msgs
            if output_msgs:
                self._agent_inv.output_messages = output_msgs
            if system_instruction:
                self._agent_inv.system_instruction = system_instruction
            if tool_defs:
                self._agent_inv.tool_definitions = tool_defs
            try:
                self._handler.stop_invoke_agent(self._agent_inv)
            except Exception as exc:
                logger.warning("CUA: stop_invoke_agent failed: %s", exc)
            self._agent_inv = None

        if self._entry_inv is not None:
            if self._first_token_ns is not None and self._run_start_ns:
                self._entry_inv.response_time_to_first_token = (
                    self._first_token_ns - self._run_start_ns
                )
            if input_msgs:
                self._entry_inv.input_messages = input_msgs
            if output_msgs:
                self._entry_inv.output_messages = output_msgs
            if system_instruction:
                self._entry_inv.system_instruction = system_instruction
            if tool_defs:
                self._entry_inv.tool_definitions = tool_defs
            try:
                self._handler.stop_entry(self._entry_inv)
            except Exception as exc:
                logger.warning("CUA: stop_entry failed: %s", exc)
            self._entry_inv = None

    async def on_run_continue(self, kwargs, old_items, new_items) -> bool:
        return True

    async def on_llm_start(self, messages):
        self._step_count += 1
        # If a previous STEP span is still open (its tools may not have finished
        # yet), close it now before opening a new one. CUA calls on_responses
        # after the LLM returns but BEFORE executing the tool calls emitted in
        # that response, so STEP must remain open until the next iteration.
        self._fail_open_tool_span()
        self._close_open_step_span()
        self._step_inv = ReactStepInvocation(round=self._step_count)
        self._step_inv.attributes[GEN_AI_FRAMEWORK] = _FRAMEWORK_VALUE
        try:
            self._handler.start_react_step(self._step_inv)
        except Exception as exc:
            logger.warning("CUA: start_react_step failed: %s", exc)
            self._step_inv = None
        return messages

    async def on_llm_end(self, output):
        return output

    async def on_responses(self, kwargs: Dict[str, Any], responses: Dict[str, Any]) -> None:
        if self._first_token_ns is None:
            self._first_token_ns = time.perf_counter_ns()
        # Stash the most recent responses so on_run_end can surface them as
        # gen_ai.output.messages on the AGENT/ENTRY spans.
        if isinstance(responses, dict):
            self._last_responses = responses
        # Do NOT close the STEP span here: tool calls emitted in this response
        # are executed after on_responses returns and must remain children of
        # the STEP. We only record the finish_reason for later use.
        if self._step_inv is not None:
            try:
                self._step_inv.finish_reason = extract_finish_reason(responses)
            except Exception:
                pass

    async def on_computer_call_start(self, item: Dict[str, Any]) -> None:
        action = (item or {}).get("action", {}) if isinstance(item, dict) else {}
        action_type = action.get("type") if isinstance(action, dict) else None
        tool_name = action_type or "computer_action"
        self._fail_open_tool_span()
        self._tool_inv = ExecuteToolInvocation(
            tool_name=tool_name,
            tool_call_id=item.get("call_id") if isinstance(item, dict) else None,
            tool_type="computer_use",
            tool_description=action_description(tool_name),
        )
        _set_optional_action_attrs(self._tool_inv.attributes, action)
        self._tool_inv.attributes[GEN_AI_FRAMEWORK] = _FRAMEWORK_VALUE
        args_str = serialize_arguments(action)
        if args_str is not None:
            self._tool_inv.tool_call_arguments = args_str
        try:
            self._handler.start_execute_tool(self._tool_inv)
        except Exception as exc:
            logger.warning("CUA: start_execute_tool (computer) failed: %s", exc)
            self._tool_inv = None

    async def on_computer_call_end(self, item: Dict[str, Any], result: Any) -> None:
        if self._tool_inv is None:
            return
        res_str = serialize_arguments(result)
        if res_str is not None:
            self._tool_inv.tool_call_result = res_str
        try:
            self._handler.stop_execute_tool(self._tool_inv)
        except Exception as exc:
            logger.warning("CUA: stop_execute_tool (computer) failed: %s", exc)
        self._tool_inv = None

    async def on_function_call_start(self, item: Dict[str, Any]) -> None:
        name = (item or {}).get("name", "function") if isinstance(item, dict) else "function"
        self._fail_open_tool_span()
        self._tool_inv = ExecuteToolInvocation(
            tool_name=name,
            tool_call_id=item.get("call_id") if isinstance(item, dict) else None,
            tool_type="function",
            tool_description=name,
        )
        self._tool_inv.attributes[GEN_AI_FRAMEWORK] = _FRAMEWORK_VALUE
        args_str = serialize_arguments(item.get("arguments") if isinstance(item, dict) else None)
        if args_str is not None:
            self._tool_inv.tool_call_arguments = args_str
        try:
            self._handler.start_execute_tool(self._tool_inv)
        except Exception as exc:
            logger.warning("CUA: start_execute_tool (function) failed: %s", exc)
            self._tool_inv = None

    async def on_function_call_end(self, item: Dict[str, Any], result: Any) -> None:
        if self._tool_inv is None:
            return
        # CUA returns FunctionCallOutput items whose `output` field carries the result string
        result_payload: Any = result
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and "output" in first:
                result_payload = first.get("output")
        res_str = serialize_arguments(result_payload)
        if res_str is not None:
            self._tool_inv.tool_call_result = res_str
        try:
            self._handler.stop_execute_tool(self._tool_inv)
        except Exception as exc:
            logger.warning("CUA: stop_execute_tool (function) failed: %s", exc)
        self._tool_inv = None

    async def on_text(self, item: Dict[str, Any]) -> None:
        pass

    async def on_api_start(self, kwargs: Dict[str, Any]) -> None:
        pass

    async def on_api_end(self, kwargs: Dict[str, Any], result: Any) -> None:
        pass

    async def on_usage(self, usage: Dict[str, Any]) -> None:
        if not isinstance(usage, dict):
            return
        try:
            self._input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            self._output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        except (TypeError, ValueError):
            pass

    async def on_screenshot(self, screenshot, name: str = "screenshot") -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fail_open_tool_span(self) -> None:
        if self._tool_inv is not None:
            try:
                self._handler.fail_execute_tool(
                    self._tool_inv,
                    Error(message="Tool call ended without completion", type=RuntimeError),
                )
            except Exception:
                pass
            self._tool_inv = None

    def _fail_open_step_span(self) -> None:
        if self._step_inv is not None:
            try:
                self._handler.fail_react_step(
                    self._step_inv,
                    Error(message="React step ended without completion", type=RuntimeError),
                )
            except Exception:
                pass
            self._step_inv = None

    def _close_open_step_span(self) -> None:
        """Stop the currently open STEP span normally (finish_reason already set)."""
        if self._step_inv is not None:
            try:
                self._handler.stop_react_step(self._step_inv)
            except Exception as exc:
                logger.warning("CUA: stop_react_step failed: %s", exc)
            self._step_inv = None

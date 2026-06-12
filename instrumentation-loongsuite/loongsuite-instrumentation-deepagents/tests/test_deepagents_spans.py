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

"""Integration tests for DeepAgents root-span classification."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class _FakeToolCallingModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-deepagents"

    @property
    def _identifying_params(self) -> dict:
        return {}

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        message = AIMessage(content="Deep agent final answer.")
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message,
                    generation_info={"finish_reason": "stop"},
                )
            ],
            llm_output={"model_name": "fake-deepagents"},
        )


def test_deepagents_root_span_is_agent(instrument, span_exporter):
    from deepagents import create_deep_agent  # noqa: PLC0415

    agent = create_deep_agent(
        model=_FakeToolCallingModel(),
        tools=[],
        system_prompt="Answer briefly.",
        name="deep_test_agent",
    )

    assert getattr(agent, "_loongsuite_react_agent") is True

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "hello"}]}
    )

    assert result["messages"][-1].content == "Deep agent final answer."

    spans = span_exporter.get_finished_spans()
    root_spans = [span for span in spans if span.parent is None]
    root_kinds = {
        span.name: span.attributes.get("gen_ai.span.kind")
        for span in root_spans
    }

    assert root_kinds == {"invoke_agent deep_test_agent": "AGENT"}
    assert not any(
        span.name == "chain deep_test_agent"
        and span.attributes.get("gen_ai.span.kind") == "CHAIN"
        for span in root_spans
    )


def test_top_level_create_deep_agent_export_is_wrapped(instrument):
    import deepagents  # noqa: PLC0415
    import deepagents.graph  # noqa: PLC0415

    assert deepagents.create_deep_agent is deepagents.graph.create_deep_agent

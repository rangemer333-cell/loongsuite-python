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

"""End-to-end tests for :class:`AGUIInstrumentor`."""

from __future__ import annotations

import importlib
from typing import Collection

import pytest

from opentelemetry.instrumentation.ag_ui import AGUIInstrumentor


class _NoDepInstrumentor(AGUIInstrumentor):
    """Test subclass that skips the ag-ui-protocol dependency check."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return ()


def test_instrument_and_uninstrument_with_stub_sdk(
    monkeypatch, agui_sdk, tracer_provider, span_exporter
):
    # Ensure fresh module-level state.
    import opentelemetry.instrumentation.ag_ui as agui_mod

    importlib.reload(agui_mod)
    instrumentor = _NoDepInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    EventEncoder = agui_sdk.EventEncoder
    RunAgentInput = agui_sdk.RunAgentInput

    # Simulate FastAPI parsing the request body.
    request_input = RunAgentInput()
    # model_post_init is called by pydantic after parsing; trigger it manually.
    RunAgentInput.model_post_init(request_input, None)

    encoder = EventEncoder()
    encoder.encode(
        type("Evt", (), {"type": "RUN_STARTED", "thread_id": "t", "run_id": "r"})()
    )
    encoder.encode(
        type("Evt", (), {"type": "RUN_FINISHED", "thread_id": "t", "run_id": "r"})()
    )

    spans = span_exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert "enter_ai_application_system" in names
    assert any("invoke_agent" in n for n in names)

    instrumentor.uninstrument()

    # After uninstrument, the ContextVar patch is restored.
    assert (
        RunAgentInput.model_post_init
        is agui_sdk.ConfiguredBaseModel.model_post_init
    )


def test_disabled_via_env_is_noop(monkeypatch, agui_sdk):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_AG_UI_ENABLED", "false")
    importlib.reload(
        importlib.import_module("opentelemetry.instrumentation.ag_ui.config")
    )
    import opentelemetry.instrumentation.ag_ui as agui_mod

    importlib.reload(agui_mod)
    instrumentor = _NoDepInstrumentor()
    instrumentor.instrument()

    encoder = agui_sdk.EventEncoder()
    encoder.encode(
        type("Evt", (), {"type": "RUN_STARTED", "thread_id": "t", "run_id": "r"})()
    )
    assert not hasattr(encoder, "_agui_span_manager")
    instrumentor.uninstrument()


@pytest.mark.parametrize("env_value", ["true", "1", "yes", "on"])
def test_env_truthy_parsing(env_value, monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_AG_UI_ENABLED", env_value)
    importlib.reload(
        importlib.import_module("opentelemetry.instrumentation.ag_ui.config")
    )
    from opentelemetry.instrumentation.ag_ui.config import AGUIConfig

    cfg = AGUIConfig.from_env()
    assert cfg.enabled is True


def test_instrumentor_idempotent_when_sdk_missing(monkeypatch, tracer_provider):
    import sys

    # Hide the stub SDK if present.
    saved = {
        k: sys.modules.get(k)
        for k in list(sys.modules)
        if k.startswith("ag_ui")
    }
    for k in list(sys.modules):
        if k.startswith("ag_ui"):
            sys.modules.pop(k, None)

    import opentelemetry.instrumentation.ag_ui as agui_mod

    importlib.reload(agui_mod)
    instrumentor = _NoDepInstrumentor()
    # Should not raise even though EventEncoder cannot be found.
    instrumentor.instrument(tracer_provider=tracer_provider)
    instrumentor.uninstrument()

    # Restore modules for subsequent tests.
    sys.modules.update({k: v for k, v in saved.items() if v is not None})


def test_stack_walking_recovers_input_when_contextvar_empty(
    monkeypatch, agui_sdk, tracer_provider, span_exporter
):
    """P1-1: when RunAgentInput.model_post_init does not fire (e.g. some
    pydantic paths in crewai/langgraph integrations), the encoder init
    must still recover the input by walking the call stack.
    """
    import sys

    # Force the stub SDK to be the one in sys.modules. A previous test may
    # have triggered real ``ag-ui-protocol`` import, which ``setdefault`` in
    # the ``agui_sdk`` fixture cannot replace.
    import types as _types

    stub_encoder_mod = _types.ModuleType("ag_ui.encoder.encoder")
    stub_encoder_mod.EventEncoder = agui_sdk.EventEncoder
    stub_types_mod = _types.ModuleType("ag_ui.core.types")
    stub_types_mod.RunAgentInput = agui_sdk.RunAgentInput
    stub_types_mod.ConfiguredBaseModel = agui_sdk.ConfiguredBaseModel
    sys.modules["ag_ui.encoder.encoder"] = stub_encoder_mod
    sys.modules["ag_ui.core.types"] = stub_types_mod

    import opentelemetry.instrumentation.ag_ui as agui_mod

    importlib.reload(agui_mod)
    instrumentor = _NoDepInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    RunAgentInput = agui_sdk.RunAgentInput
    EventEncoder = agui_sdk.EventEncoder

    # Construct an input WITHOUT triggering model_post_init (simulates the
    # crewai/langgraph integration path where the ContextVar never gets set).
    request_input = RunAgentInput.__new__(RunAgentInput)
    request_input.messages = [
        type("M", (), {"role": "user", "content": "hi"})()
    ]
    request_input.tools = []
    request_input.forwarded_props = {}

    # Create the encoder in a frame that has ``input_data`` as a local,
    # mirroring the FastAPI endpoint pattern used by all integrations.
    def endpoint_like(input_data):
        encoder = EventEncoder()
        encoder.encode(
            type("Evt", (), {"type": "RUN_STARTED", "thread_id": "t", "run_id": "r"})()
        )
        encoder.encode(
            type("Evt", (), {"type": "RUN_FINISHED", "thread_id": "t", "run_id": "r"})()
        )
        return encoder

    endpoint_like(request_input)

    instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    entry = next(s for s in spans if s.name == "enter_ai_application_system")
    # If stack walking recovered the input, gen_ai.input.messages must be set
    # because capture_content defaults to True under the conftest env.
    assert "gen_ai.input.messages" in entry.attributes
    assert "hi" in entry.attributes["gen_ai.input.messages"]

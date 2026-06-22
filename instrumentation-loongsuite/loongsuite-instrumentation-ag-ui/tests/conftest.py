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

"""Shared test fixtures for the AG-UI instrumentation tests."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(autouse=True)
def _content_capture_env(monkeypatch):
    """Enable util-genai content capture for the duration of each test.

    The util-genai layer only writes gen_ai.input.messages /
    gen_ai.tool.call.arguments / ... when both experimental mode is opted-in
    and the capture mode is SPAN-only. Tests that exercise capture_content
    rely on these env vars so the AG-UI instrumentation and the util-genai
    writer stay in sync.
    """
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    # Re-import utils so the module-level caches pick up the env values.
    import importlib

    import opentelemetry.util.genai.utils as genai_utils

    importlib.reload(genai_utils)


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    return exporter


@pytest.fixture
def tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    yield provider


@pytest.fixture
def handler(tracer_provider):
    from opentelemetry.util.genai.extended_handler import (
        ExtendedTelemetryHandler,
    )

    return ExtendedTelemetryHandler(tracer_provider=tracer_provider)


class _StubEvent:
    """Minimal stand-in for AG-UI typed events."""

    def __init__(self, type: str, **fields: Any) -> None:
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


@pytest.fixture
def stub_event():
    return _StubEvent


@pytest.fixture
def agui_sdk(monkeypatch):
    """Provide a synthetic ``ag_ui`` package on sys.path for tests.

    The instrumentation patches classes inside ``ag_ui.encoder.encoder`` and
    ``ag_ui.core.types``. We install lightweight stub modules so the tests do
    not require the real SDK at collection time.
    """
    package = types.ModuleType("ag_ui")
    encoder_pkg = types.ModuleType("ag_ui.encoder")
    encoder_mod = types.ModuleType("ag_ui.encoder.encoder")
    core_pkg = types.ModuleType("ag_ui.core")
    events_mod = types.ModuleType("ag_ui.core.events")
    types_mod = types.ModuleType("ag_ui.core.types")

    class EventEncoder:
        def __init__(self, accept: str | None = None) -> None:
            self.accept = accept

        def encode(self, event: Any) -> str:
            return f"data: {getattr(event, 'type', '')}\n\n"

    class ConfiguredBaseModel:
        def model_post_init(self, __context: Any) -> None:  # noqa: D401
            pass

    class RunAgentInput(ConfiguredBaseModel):
        pass

    class EventType:
        RUN_STARTED = "RUN_STARTED"
        RUN_FINISHED = "RUN_FINISHED"
        RUN_ERROR = "RUN_ERROR"
        STEP_STARTED = "STEP_STARTED"
        STEP_FINISHED = "STEP_FINISHED"
        TOOL_CALL_START = "TOOL_CALL_START"
        TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
        TOOL_CALL_END = "TOOL_CALL_END"
        TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
        TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
        MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"

    encoder_mod.EventEncoder = EventEncoder
    types_mod.ConfiguredBaseModel = ConfiguredBaseModel
    types_mod.RunAgentInput = RunAgentInput
    events_mod.EventType = EventType

    package.__path__ = []  # mark as package
    encoder_pkg.__path__ = []
    core_pkg.__path__ = []

    sys.modules.setdefault("ag_ui", package)
    sys.modules.setdefault("ag_ui.encoder", encoder_pkg)
    sys.modules.setdefault("ag_ui.encoder.encoder", encoder_mod)
    sys.modules.setdefault("ag_ui.core", core_pkg)
    sys.modules.setdefault("ag_ui.core.events", events_mod)
    sys.modules.setdefault("ag_ui.core.types", types_mod)

    yield types.SimpleNamespace(
        EventEncoder=EventEncoder,
        RunAgentInput=RunAgentInput,
        ConfiguredBaseModel=ConfiguredBaseModel,
        EventType=EventType,
    )

    for name in (
        "ag_ui",
        "ag_ui.encoder",
        "ag_ui.encoder.encoder",
        "ag_ui.core",
        "ag_ui.core.events",
        "ag_ui.core.types",
    ):
        sys.modules.pop(name, None)

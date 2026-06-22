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

"""Shared test configuration for CUA instrumentation unit tests."""

import os

import pytest

# Opt into the experimental GenAI semantic conventions before importing
# opentelemetry-instrumentation internals. The instrumentor module imports
# happen lazily inside fixtures, but the env vars must be set before
# opentelemetry.instrumentation._semconv is imported anywhere.
os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")
# Default to capturing content in tests so we exercise that code path
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")

from opentelemetry.instrumentation._semconv import (  # noqa: E402
    OTEL_SEMCONV_STABILITY_OPT_IN,
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.util.genai.environment_variables import (  # noqa: E402
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler  # noqa: E402


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="handler")
def fixture_handler(tracer_provider):
    return ExtendedTelemetryHandler(tracer_provider=tracer_provider)


@pytest.fixture(scope="function", name="instrument_no_content")
def fixture_instrument_no_content(tracer_provider, monkeypatch):
    """Reset stability + content capture mode so a fresh read picks up NO_CONTENT."""
    _OpenTelemetrySemanticConventionStability._initialized = False
    monkeypatch.setenv(OTEL_SEMCONV_STABILITY_OPT_IN, "gen_ai_latest_experimental")
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "NO_CONTENT")
    yield tracer_provider
    _OpenTelemetrySemanticConventionStability._initialized = False


@pytest.fixture(scope="function", name="instrument_with_content")
def fixture_instrument_with_content(tracer_provider, monkeypatch):
    _OpenTelemetrySemanticConventionStability._initialized = False
    monkeypatch.setenv(OTEL_SEMCONV_STABILITY_OPT_IN, "gen_ai_latest_experimental")
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_ONLY")
    yield tracer_provider
    _OpenTelemetrySemanticConventionStability._initialized = False

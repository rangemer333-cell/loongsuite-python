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

"""Unit tests for the Sandbox monkey-patch wrappers.

We exercise the wrappers directly against stub functions instead of the
real cua_sandbox.Sandbox class — this keeps the test environment free of
heavy CUA dependencies while still verifying that the wrappers create
TASK spans with the right attributes.
"""

import asyncio
import json

import pytest

from opentelemetry.instrumentation.cua.sandbox_patch import (
    _wrap_connect,
    _wrap_create,
    _wrap_destroy,
    _wrap_disconnect,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import get_tracer


class _FakeRuntime:
    pass


class _FakeSandbox:
    def __init__(self, name="sb-1"):
        self.name = name
        self._runtime = _FakeRuntime()
        self._transport = None


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_tracer():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = get_tracer("test", "0", tracer_provider=provider)
    return tracer, exporter


async def _async_create(image, *, name=None, **kwargs):
    return _FakeSandbox(name=name or "auto")


async def _async_destroy(*args, **kwargs):
    return None


async def _async_connect(name=None, **kwargs):
    return _FakeSandbox(name=name or "auto")


async def _async_disconnect(*args, **kwargs):
    return None


def test_create_emits_task_span_with_sandbox_attrs():
    tracer, exporter = _make_tracer()
    wrapped = _wrap_create(tracer)

    async def scenario():
        return await wrapped(_async_create, None, ("ubuntu:24.04",), {
            "name": "sb-prod",
            "local": True,
            "cpu": 4,
            "memory_mb": 2048,
            "region": "us-west-2",
        })

    result = _run(scenario())
    assert isinstance(result, _FakeSandbox)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "run_task sandbox.create"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.span.kind"] == "TASK"
    assert attrs["gen_ai.operation.name"] == "run_task"
    assert attrs["cua.sandbox.image"] == "ubuntu:24.04"
    assert attrs["cua.sandbox.name"] == "sb-prod"
    assert attrs["cua.sandbox.local"] is True
    assert attrs["cua.sandbox.cpu"] == 4
    assert attrs["cua.sandbox.memory_mb"] == 2048
    assert attrs["cua.sandbox.region"] == "us-west-2"
    assert attrs["cua.sandbox.runtime"] == "_FakeRuntime"
    # Regression for review problem 2: standard TASK input/output payloads
    assert attrs["input.mime_type"] == "application/json"
    assert attrs["output.mime_type"] == "application/json"
    input_payload = json.loads(attrs["input.value"])
    assert input_payload["image"] == "ubuntu:24.04"
    assert input_payload["name"] == "sb-prod"
    assert input_payload["local"] is True
    assert input_payload["cpu"] == 4
    assert input_payload["memory_mb"] == 2048
    assert input_payload["region"] == "us-west-2"
    output_payload = json.loads(attrs["output.value"])
    assert output_payload["name"] == "sb-prod"
    assert output_payload["runtime"] == "_FakeRuntime"


def test_destroy_emits_task_span():
    tracer, exporter = _make_tracer()
    wrapped = _wrap_destroy(tracer)
    sb = _FakeSandbox(name="sb-destroy")

    async def scenario():
        return await wrapped(_async_destroy, sb, (), {})

    _run(scenario())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "run_task sandbox.destroy"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.span.kind"] == "TASK"
    assert attrs["gen_ai.operation.name"] == "run_task"
    assert attrs["cua.sandbox.name"] == "sb-destroy"
    # destroy has no output, but should still carry input payload
    assert attrs["input.mime_type"] == "application/json"
    input_payload = json.loads(attrs["input.value"])
    assert input_payload["name"] == "sb-destroy"
    assert "output.value" not in attrs
    assert "output.mime_type" not in attrs


def test_connect_emits_task_span():
    tracer, exporter = _make_tracer()
    wrapped = _wrap_connect(tracer)

    async def scenario():
        # Sandbox.connect(name, *, ...)
        return await wrapped(_async_connect, None, ("existing-sb",), {"local": False})

    result = _run(scenario())
    assert isinstance(result, _FakeSandbox)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "run_task sandbox.connect"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.span.kind"] == "TASK"
    assert attrs["cua.sandbox.name"] == "existing-sb"
    assert attrs["cua.sandbox.local"] is False
    # connect emits both input and output payloads
    assert attrs["input.mime_type"] == "application/json"
    assert attrs["output.mime_type"] == "application/json"
    input_payload = json.loads(attrs["input.value"])
    assert input_payload["name"] == "existing-sb"
    assert input_payload["local"] is False
    output_payload = json.loads(attrs["output.value"])
    assert output_payload["name"] == "existing-sb"
    assert output_payload["runtime"] == "_FakeRuntime"


def test_disconnect_emits_task_span():
    tracer, exporter = _make_tracer()
    wrapped = _wrap_disconnect(tracer)
    sb = _FakeSandbox(name="sb-disconnect")

    async def scenario():
        return await wrapped(_async_disconnect, sb, (), {})

    _run(scenario())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "run_task sandbox.disconnect"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.span.kind"] == "TASK"
    assert attrs["cua.sandbox.name"] == "sb-disconnect"
    # disconnect carries input payload, no output
    assert attrs["input.mime_type"] == "application/json"
    input_payload = json.loads(attrs["input.value"])
    assert input_payload["name"] == "sb-disconnect"
    assert "output.value" not in attrs
    assert "output.mime_type" not in attrs


def test_create_failure_marks_error():
    tracer, exporter = _make_tracer()
    wrapped = _wrap_create(tracer)

    async def boom(image, *, name=None, **kwargs):
        raise RuntimeError("sandbox create failed")

    async def scenario():
        with pytest.raises(RuntimeError):
            await wrapped(boom, None, ("ubuntu",), {"name": "fail-sb"})

    _run(scenario())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.is_ok is False
    attrs = dict(span.attributes or {})
    assert attrs["cua.sandbox.image"] == "ubuntu"
    assert attrs["cua.sandbox.name"] == "fail-sb"
    # Even on failure, the input payload must be present (output is omitted)
    assert attrs["input.mime_type"] == "application/json"
    input_payload = json.loads(attrs["input.value"])
    assert input_payload["image"] == "ubuntu"
    assert input_payload["name"] == "fail-sb"
    assert "output.value" not in attrs
    assert "output.mime_type" not in attrs


def test_task_span_input_output_payloads_regression():
    """Regression for review problem 2/7c: every TASK span must carry
    ``input.value``/``input.mime_type`` (all four wrappers) and
    ``output.value``/``output.mime_type`` (create/connect only), as JSON
    with ``application/json`` mime type — aligned with ``execute.md`` §4.7.
    """
    tracer, exporter = _make_tracer()

    # create — input + output
    create_wrapped = _wrap_create(tracer)

    async def create_scenario():
        return await create_wrapped(_async_create, None, ("ubuntu:24.04",), {
            "name": "sb-create",
            "local": True,
            "cpu": 2,
            "memory_mb": 1024,
            "region": "us-east-1",
        })

    _run(create_scenario())
    span = exporter.get_finished_spans()[-1]
    attrs = dict(span.attributes or {})
    assert attrs["input.mime_type"] == "application/json"
    assert attrs["output.mime_type"] == "application/json"
    create_input = json.loads(attrs["input.value"])
    assert create_input == {
        "image": "ubuntu:24.04",
        "name": "sb-create",
        "local": True,
        "cpu": 2,
        "memory_mb": 1024,
        "region": "us-east-1",
    }
    create_output = json.loads(attrs["output.value"])
    assert create_output["name"] == "sb-create"
    assert create_output["runtime"] == "_FakeRuntime"

    exporter.clear()

    # connect — input + output, name positional
    connect_wrapped = _wrap_connect(tracer)

    async def connect_scenario():
        return await connect_wrapped(_async_connect, None, ("sb-connect",), {"local": False})

    _run(connect_scenario())
    span = exporter.get_finished_spans()[-1]
    attrs = dict(span.attributes or {})
    assert attrs["input.mime_type"] == "application/json"
    assert attrs["output.mime_type"] == "application/json"
    connect_input = json.loads(attrs["input.value"])
    assert connect_input == {"name": "sb-connect", "local": False}
    connect_output = json.loads(attrs["output.value"])
    assert connect_output["name"] == "sb-connect"

    exporter.clear()

    # destroy — input only
    destroy_wrapped = _wrap_destroy(tracer)
    sb = _FakeSandbox(name="sb-destroy")

    async def destroy_scenario():
        return await destroy_wrapped(_async_destroy, sb, (), {})

    _run(destroy_scenario())
    span = exporter.get_finished_spans()[-1]
    attrs = dict(span.attributes or {})
    assert attrs["input.mime_type"] == "application/json"
    assert json.loads(attrs["input.value"]) == {"name": "sb-destroy"}
    assert "output.value" not in attrs
    assert "output.mime_type" not in attrs

    exporter.clear()

    # disconnect — input only
    disconnect_wrapped = _wrap_disconnect(tracer)
    sb = _FakeSandbox(name="sb-disconnect")

    async def disconnect_scenario():
        return await disconnect_wrapped(_async_disconnect, sb, (), {})

    _run(disconnect_scenario())
    span = exporter.get_finished_spans()[-1]
    attrs = dict(span.attributes or {})
    assert attrs["input.mime_type"] == "application/json"
    assert json.loads(attrs["input.value"]) == {"name": "sb-disconnect"}
    assert "output.value" not in attrs
    assert "output.mime_type" not in attrs

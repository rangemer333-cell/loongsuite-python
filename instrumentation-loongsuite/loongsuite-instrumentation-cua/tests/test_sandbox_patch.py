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

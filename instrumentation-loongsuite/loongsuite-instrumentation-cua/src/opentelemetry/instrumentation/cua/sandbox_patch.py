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

"""Monkey patches for CUA Sandbox lifecycle methods (TASK spans)."""

import logging
from typing import Any, Callable, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode, set_span_in_context
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

logger = logging.getLogger(__name__)


def _span_kind_task() -> str:
    return "TASK"


def _sandbox_attrs_before(args: tuple, kwargs: dict, *, name_positional: bool = False) -> dict:
    """Build sandbox.* attributes from create/ephemeral/connect args."""
    attrs: dict = {}
    # Sandbox.create(image, *, name=, local=, cpu=, memory_mb=, region=...)
    # Sandbox.connect(name, *, local=, region=...) — name is positional
    image = args[0] if args else kwargs.get("image")
    name = kwargs.get("name")
    if name is None and name_positional and args:
        name = args[0]
    local = kwargs.get("local", False)
    cpu = kwargs.get("cpu")
    memory_mb = kwargs.get("memory_mb")
    region = kwargs.get("region", "us-east-1")
    if image is not None:
        attrs["cua.sandbox.image"] = str(image)
    if name is not None:
        attrs["cua.sandbox.name"] = name
    attrs["cua.sandbox.local"] = bool(local)
    if cpu is not None:
        attrs["cua.sandbox.cpu"] = cpu
    if memory_mb is not None:
        attrs["cua.sandbox.memory_mb"] = memory_mb
    if region is not None:
        attrs["cua.sandbox.region"] = region
    return attrs


def _set_runtime_attrs(span: Any, sandbox: Any) -> None:
    if sandbox is None:
        return
    name = getattr(sandbox, "name", None)
    if name:
        span.set_attribute("cua.sandbox.name", name)
    runtime = getattr(sandbox, "_runtime", None)
    if runtime is not None:
        span.set_attribute("cua.sandbox.runtime", type(runtime).__name__)
    transport = getattr(sandbox, "_transport", None)
    if transport is not None:
        transport_cls = type(transport).__name__
        span.set_attribute("cua.sandbox.transport", transport_cls)


def _wrap_create(tracer: trace.Tracer) -> Callable:
    async def wrapper(wrapped, instance, args, kwargs):
        attrs = _sandbox_attrs_before(args, kwargs)
        span = tracer.start_span(
            name="run_task sandbox.create",
            kind=SpanKind.INTERNAL,
            attributes={
                GEN_AI_SPAN_KIND: _span_kind_task(),
                "gen_ai.operation.name": "run_task",
                **attrs,
            },
        )
        ctx = otel_context.attach(set_span_in_context(span))
        try:
            result = await wrapped(*args, **kwargs)
            _set_runtime_attrs(span, result)
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            otel_context.detach(ctx)
            span.end()

    return wrapper


def _wrap_destroy(tracer: trace.Tracer) -> Callable:
    async def wrapper(wrapped, instance, args, kwargs):
        name = getattr(instance, "name", None) if instance is not None else None
        span = tracer.start_span(
            name="run_task sandbox.destroy",
            kind=SpanKind.INTERNAL,
            attributes={
                GEN_AI_SPAN_KIND: _span_kind_task(),
                "gen_ai.operation.name": "run_task",
            },
        )
        if name:
            span.set_attribute("cua.sandbox.name", name)
        ctx = otel_context.attach(set_span_in_context(span))
        try:
            result = await wrapped(*args, **kwargs)
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            otel_context.detach(ctx)
            span.end()

    return wrapper


def _wrap_connect(tracer: trace.Tracer) -> Callable:
    """Sandbox.connect returns a _ConnectResult awaiting the actual connect."""
    async def wrapper(wrapped, instance, args, kwargs):
        attrs = _sandbox_attrs_before(args, kwargs, name_positional=True)
        span = tracer.start_span(
            name="run_task sandbox.connect",
            kind=SpanKind.INTERNAL,
            attributes={
                GEN_AI_SPAN_KIND: _span_kind_task(),
                "gen_ai.operation.name": "run_task",
                **attrs,
            },
        )
        ctx = otel_context.attach(set_span_in_context(span))
        try:
            result = await wrapped(*args, **kwargs)
            _set_runtime_attrs(span, result)
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            otel_context.detach(ctx)
            span.end()

    return wrapper


def _wrap_disconnect(tracer: trace.Tracer) -> Callable:
    async def wrapper(wrapped, instance, args, kwargs):
        name = getattr(instance, "name", None) if instance is not None else None
        span = tracer.start_span(
            name="run_task sandbox.disconnect",
            kind=SpanKind.INTERNAL,
            attributes={
                GEN_AI_SPAN_KIND: _span_kind_task(),
                "gen_ai.operation.name": "run_task",
            },
        )
        if name:
            span.set_attribute("cua.sandbox.name", name)
        ctx = otel_context.attach(set_span_in_context(span))
        try:
            result = await wrapped(*args, **kwargs)
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            otel_context.detach(ctx)
            span.end()

    return wrapper

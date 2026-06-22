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

import json
import logging
from typing import Any, Callable, List, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode, Link, set_span_in_context
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_SPAN_KIND,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

logger = logging.getLogger(__name__)

_INPUT_MIME_TYPE = "application/json"
_OUTPUT_MIME_TYPE = "application/json"

GEN_AI_FRAMEWORK = "gen_ai.framework"
_FRAMEWORK_VALUE = "cua"


def _entry_span_link() -> List[Link]:
    """Build a single-element ``links`` list pointing at the most recent CUA
    ENTRY span context (if any), so the ``run_task sandbox.destroy`` TASK span
    is no longer an orphan trace — it cross-references the agent run it
    belongs to (per verification report 7ca3c1df P2.4).

    Returns an empty list when no ENTRY has run yet (e.g. example 01's
    standalone sandbox lifecycle test, or the destroy-before-run edge case),
    in which case the destroy span stays a root span as before.
    """
    try:
        # Lazy import to avoid any module-load ordering concerns.
        from opentelemetry.instrumentation.cua.callback import (
            _latest_entry_span_context,
        )
    except Exception:
        return []
    try:
        ctx = _latest_entry_span_context.get()
    except Exception:
        return []
    if ctx is None:
        return []
    if not getattr(ctx, "is_valid", False):
        return []
    return [Link(ctx)]


def _apply_framework_attr(attrs: dict) -> None:
    """Stamp ``gen_ai.framework=cua`` on a sandbox TASK span attribute dict."""
    attrs[GEN_AI_FRAMEWORK] = _FRAMEWORK_VALUE


def _span_kind_task() -> str:
    return "TASK"


def _sandbox_input_payload(args: tuple, kwargs: dict, *, name_positional: bool = False) -> dict:
    """Build the ``input.value`` JSON payload from create/connect args.

    Aligned with ``execute.md`` §4.7: only fields with a non-None value are
    included so the JSON stays small and deterministic.
    """
    payload: dict = {}
    name = kwargs.get("name")
    if name is None and name_positional and args:
        name = args[0]
    # ``image`` is the first positional arg of Sandbox.create; for
    # Sandbox.connect the first positional arg is the sandbox name and
    # there is no image.
    image = args[0] if args and not name_positional else kwargs.get("image")
    local = kwargs.get("local", False)
    cpu = kwargs.get("cpu")
    memory_mb = kwargs.get("memory_mb")
    region = kwargs.get("region")
    if image is not None:
        payload["image"] = str(image)
    if name is not None:
        payload["name"] = name
    payload["local"] = bool(local)
    if cpu is not None:
        payload["cpu"] = cpu
    if memory_mb is not None:
        payload["memory_mb"] = memory_mb
    if region is not None:
        payload["region"] = region
    return payload


def _sandbox_output_payload(sandbox: Any) -> dict:
    """Build the ``output.value`` JSON payload from a create/connect result."""
    if sandbox is None:
        return {}
    payload: dict = {}
    name = getattr(sandbox, "name", None)
    if name:
        payload["name"] = name
    runtime = getattr(sandbox, "_runtime", None)
    if runtime is not None:
        payload["runtime"] = type(runtime).__name__
    transport = getattr(sandbox, "_transport", None)
    if transport is not None:
        payload["transport"] = type(transport).__name__
    return payload


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


def _start_task_span(
    tracer: trace.Tracer,
    name: str,
    attrs: dict,
    input_payload: dict,
    links: Optional[List[Link]] = None,
) -> Any:
    """Start a TASK span with the standard ``input.value``/``input.mime_type``.

    ``links`` lets callers attach OTel links to related spans (e.g. linking a
    ``run_task sandbox.destroy`` span to the most recent ENTRY span context so
    the destroy TASK is no longer an orphan trace — see P2.4).
    """
    _apply_framework_attr(attrs)
    final_attrs = {
        GEN_AI_SPAN_KIND: _span_kind_task(),
        "gen_ai.operation.name": "run_task",
        "input.mime_type": _INPUT_MIME_TYPE,
        **attrs,
    }
    if input_payload:
        final_attrs["input.value"] = json.dumps(input_payload, sort_keys=True, default=str)
    return tracer.start_span(
        name=name,
        kind=SpanKind.INTERNAL,
        attributes=final_attrs,
        links=links or [],
    )


def _set_output_payload(span: Any, sandbox: Any) -> None:
    """Attach ``output.value``/``output.mime_type`` for create/connect results."""
    payload = _sandbox_output_payload(sandbox)
    if payload:
        span.set_attribute("output.value", json.dumps(payload, sort_keys=True, default=str))
        span.set_attribute("output.mime_type", _OUTPUT_MIME_TYPE)


def _wrap_create(tracer: trace.Tracer) -> Callable:
    async def wrapper(wrapped, instance, args, kwargs):
        attrs = _sandbox_attrs_before(args, kwargs)
        input_payload = _sandbox_input_payload(args, kwargs)
        span = _start_task_span(
            tracer, "run_task sandbox.create", attrs, input_payload
        )
        ctx = otel_context.attach(set_span_in_context(span))
        try:
            result = await wrapped(*args, **kwargs)
            _set_runtime_attrs(span, result)
            _set_output_payload(span, result)
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            otel_context.detach(ctx)
            span.end()

    return wrapper


def _sandbox_destroy_attrs_from_instance(instance: Any) -> dict:
    """Build cua.sandbox.* attributes from a Sandbox instance at destroy time.

    ``Sandbox.create`` records image/local/cpu/memory_mb/region at creation;
    ``destroy`` only has the live instance. We mirror those fields back when
    the instance still exposes them so destroy spans carry the same context
    (per execute.md §4.7 — destroy span should mirror create's cua.sandbox.*).
    """
    attrs: dict = {}
    if instance is None:
        return attrs
    name = getattr(instance, "name", None)
    if name:
        attrs["cua.sandbox.name"] = name
    image = getattr(instance, "image", None) or getattr(instance, "_image", None)
    if image is not None:
        attrs["cua.sandbox.image"] = str(image)
    local = getattr(instance, "local", None)
    if local is not None:
        attrs["cua.sandbox.local"] = bool(local)
    cpu = getattr(instance, "cpu", None)
    if cpu is not None:
        attrs["cua.sandbox.cpu"] = cpu
    memory_mb = getattr(instance, "memory_mb", None)
    if memory_mb is not None:
        attrs["cua.sandbox.memory_mb"] = memory_mb
    region = getattr(instance, "region", None)
    if region is not None:
        attrs["cua.sandbox.region"] = region
    runtime = getattr(instance, "_runtime", None)
    if runtime is not None:
        attrs["cua.sandbox.runtime"] = type(runtime).__name__
    transport = getattr(instance, "_transport", None)
    if transport is not None:
        attrs["cua.sandbox.transport"] = type(transport).__name__
    return attrs


def _wrap_destroy(tracer: trace.Tracer) -> Callable:
    async def wrapper(wrapped, instance, args, kwargs):
        name = getattr(instance, "name", None) if instance is not None else None
        input_payload: dict = {}
        if name:
            input_payload["name"] = name
        attrs = _sandbox_destroy_attrs_from_instance(instance)
        # Link to the most recent ENTRY span context so this destroy TASK
        # span is no longer an orphan trace (P2.4). No-op when no ENTRY has
        # run (e.g. example 01 sandbox-only lifecycle).
        links = _entry_span_link()
        span = _start_task_span(
            tracer, "run_task sandbox.destroy", attrs, input_payload, links=links
        )
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


def _wrap_ephemeral(tracer: trace.Tracer) -> Callable:
    """Wrap ``Sandbox.ephemeral`` (an ``@asynccontextmanager`` classmethod).

    ``ephemeral`` calls ``Sandbox._create`` directly inside its async generator,
    bypassing the patched ``Sandbox.create``. We wrap the returned context
    manager so that ``__aenter__`` (which runs ``_create``) emits a
    ``run_task sandbox.create`` TASK span with the same attributes as the
    ``create`` wrapper. The ``destroy`` span is emitted separately by the
    existing ``_wrap_destroy`` patch when ``ephemeral.__aexit__`` calls
    ``sb.destroy()`` — so we end the create span before the body runs to keep
    both TASK spans as independent roots (per ``execute.md`` §4.7).
    """

    def wrapper(wrapped, instance, args, kwargs):
        cm = wrapped(*args, **kwargs)
        attrs = _sandbox_attrs_before(args, kwargs)
        input_payload = _sandbox_input_payload(args, kwargs)

        class _TracedEphemeral:
            async def __aenter__(self_inner):
                span = _start_task_span(
                    tracer, "run_task sandbox.create", attrs, input_payload
                )
                ctx = otel_context.attach(set_span_in_context(span))
                self_inner._span = span
                self_inner._ctx = ctx
                try:
                    sb = await cm.__aenter__()
                    _set_runtime_attrs(span, sb)
                    _set_output_payload(span, sb)
                    return sb
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                finally:
                    otel_context.detach(ctx)
                    span.end()

            async def __aexit__(self_inner, exc_type, exc, tb):
                return await cm.__aexit__(exc_type, exc, tb)

        return _TracedEphemeral()

    return wrapper


def _wrap_connect(tracer: trace.Tracer) -> Callable:
    """Sandbox.connect returns a _ConnectResult awaiting the actual connect."""
    async def wrapper(wrapped, instance, args, kwargs):
        attrs = _sandbox_attrs_before(args, kwargs, name_positional=True)
        input_payload = _sandbox_input_payload(args, kwargs, name_positional=True)
        span = _start_task_span(
            tracer, "run_task sandbox.connect", attrs, input_payload
        )
        ctx = otel_context.attach(set_span_in_context(span))
        try:
            result = await wrapped(*args, **kwargs)
            _set_runtime_attrs(span, result)
            _set_output_payload(span, result)
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
        input_payload = {"name": name} if name else {}
        attrs: dict = {}
        if name:
            attrs["cua.sandbox.name"] = name
        span = _start_task_span(tracer, "run_task sandbox.disconnect", attrs, input_payload)
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

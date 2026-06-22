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

"""
OpenTelemetry CUA (Computer-Use Agent) Instrumentation
=====================================================

Auto-instrumentation for the CUA framework (https://github.com/trycua/cua).

The instrumentation:

1. Patches ``ComputerAgent.__init__`` to attach an :class:`ArmsCuaCallback`
   to every agent's callback list. The callback emits ENTRY / AGENT / STEP /
   TOOL spans using the ARMS GenAI semantic conventions via the
   ``ExtendedTelemetryHandler`` helper from ``opentelemetry-util-genai``.
2. Patches ``Sandbox.create / destroy / connect / disconnect`` to emit TASK
   spans covering sandbox lifecycle operations.

LLM spans are not emitted by this instrumentation; CUA drives LLM calls
through ``litellm`` and is already covered by the LiteLLM/OpenAI/Anthropic
instrumentation, avoiding double spans and attribute conflicts.

Usage::

    from opentelemetry.instrumentation.cua import CuaInstrumentor

    CuaInstrumentor().instrument()

    # use CUA normally
    from cua_agent import ComputerAgent
    agent = ComputerAgent(model="anthropic/claude-sonnet-4-5-20250929")
    await agent.run(messages)
"""

import logging
import os
from typing import Any, Collection, Optional

from wrapt import wrap_function_wrapper

from opentelemetry import baggage, context as otel_context
from opentelemetry.instrumentation.cua.package import _instruments
from opentelemetry.instrumentation.cua.version import __version__
from opentelemetry.instrumentation.cua.callback import (
    ArmsCuaCallback,
    GEN_AI_FRAMEWORK,
    _FRAMEWORK_VALUE,
)
from opentelemetry.instrumentation.cua.sandbox_patch import (
    _apply_framework_attr,
    _wrap_connect,
    _wrap_destroy,
    _wrap_disconnect,
    _wrap_create,
    _wrap_ephemeral,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

try:  # ``SpanProcessor`` lives in the SDK; guard for API-only environments.
    from opentelemetry.sdk.trace import SpanProcessor
except ImportError:  # pragma: no cover - exercised only without the SDK.
    class SpanProcessor:  # type: ignore[no-redef]
        """Fallback base class so module import never hard-fails without
        the SDK. Real instrumentation requires ``opentelemetry-sdk``."""

logger = logging.getLogger(__name__)

_CUA_AGENT_MODULE = "cua_agent.agent"
_CUA_AGENT_CLASS = "ComputerAgent"
_CUA_SANDBOX_MODULE = "cua_sandbox.sandbox"
_CUA_SANDBOX_CLASS = "Sandbox"

_ARMS_CALLBACK_ATTR = "_arms_cua_callback"

# Resource attribute identifying the application as a GenAI app, per
# ``/home/admin/semantic-conventions/arms_docs/trace/gen-ai.md`` §"应用特征".
_ARMS_SERVICE_FEATURE_KEY = "acs.arms.service.feature"
_ARMS_SERVICE_FEATURE_VALUE = "genai_app"


class _CuaFrameworkSpanProcessor(SpanProcessor):
    """SpanProcessor that stamps ``gen_ai.framework`` from OTel baggage onto
    spans that don't already carry the attribute.

    The CUA callback sets ``gen_ai.framework=cua`` directly on ENTRY/AGENT/
    STEP/TOOL/TASK spans via ``invocation.attributes``. LLM spans, however,
    are emitted by an external instrumentor (``loongsuite-instrumentation-
    litellm`` / ``opentelemetry-instrumentation-anthropic`` /
    ``opentelemetry-instrumentation-openai-v2``) which doesn't know about the
    CUA framework. The CUA callback attaches the framework to the active OTel
    context as baggage in ``on_run_start``; this processor reads that baggage
    on every span ``on_start`` and copies the framework attribute onto the
    span if it isn't already present.

    See verification report 7ca3c1df P2.3.
    """

    def on_start(self, span: Any, parent_context: Optional[Any] = None) -> None:
        try:
            ctx = parent_context if parent_context is not None else otel_context.get_current()
            # ``baggage.get_value`` in some opentelemetry-api builds is a
            # re-export of ``context.get_value`` (raw context lookup), which
            # does NOT see baggage entries stored under the baggage key.
            # ``baggage.get_all`` reads the baggage sub-dict correctly, so we
            # use it and then pull the framework key.
            framework = baggage.get_all(ctx).get(GEN_AI_FRAMEWORK)
        except Exception:
            return
        if framework is None:
            return
        # Avoid double-stamping if the cua callback (or any other caller)
        # already set the attribute. ``Span.attributes`` is generally empty at
        # ``on_start`` for SDK spans (attributes are usually applied via
        # ``set_attributes`` after creation), but we check defensively.
        try:
            existing = getattr(span, "attributes", None) or {}
            if GEN_AI_FRAMEWORK in existing:
                return
        except Exception:
            pass
        try:
            span.set_attribute(GEN_AI_FRAMEWORK, framework)
        except Exception:
            pass

    def on_end(self, span: Any) -> None:  # pylint: disable=no-self-use,unused-argument
        pass

    def _on_ending(self, span: Any) -> None:  # pylint: disable=no-self-use,unused-argument
        # opentelemetry-sdk>=1.42 introduced ``_on_ending`` and calls it
        # from ``Span.end()`` before ``on_end``. The ``SpanProcessor`` ABC
        # provides a no-op there, but we redefine it explicitly so the
        # processor also works on SDK<1.42 (where the base lacks the
        # method entirely) — see verification 5173946b.
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pylint: disable=unused-argument
        return True


def _merge_resource_attribute(tracer_provider: Any) -> None:
    """Best-effort merge ``acs.arms.service.feature=genai_app`` onto the
    tracer provider's resource.

    Mutates ``tracer_provider._resource`` via ``Resource.merge``. Uses a
    bare ``Resource({...})`` (no ``.create``) so the SDK defaults such as
    ``service.name=unknown_service`` do NOT override the user's existing
    resource. Safe no-op when the provider does not expose ``_resource``.
    """
    if tracer_provider is None:
        return
    try:
        from opentelemetry.sdk.resources import Resource
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("CUA: opentelemetry.sdk.resources unavailable: %s", exc)
        return
    try:
        current = getattr(tracer_provider, "_resource", None) or getattr(
            tracer_provider, "resource", None
        )
        if current is None:
            return
        existing = getattr(current, "attributes", {}) or {}
        if existing.get(_ARMS_SERVICE_FEATURE_KEY) == _ARMS_SERVICE_FEATURE_VALUE:
            return
        merged = current.merge(Resource({_ARMS_SERVICE_FEATURE_KEY: _ARMS_SERVICE_FEATURE_VALUE}))
        tracer_provider._resource = merged
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("CUA: failed to merge resource attribute: %s", exc)


# Tracks whether the framework span processor has been registered on a
# given tracer provider, so re-instrumentation doesn't pile up duplicates.
_framework_processor_providers: "set[int]" = set()


def _register_framework_span_processor(tracer_provider: Any) -> None:
    """Attach ``_CuaFrameworkSpanProcessor`` to ``tracer_provider`` (or the
    global provider when ``tracer_provider`` is None). Idempotent per
    provider object; silently skips when the SDK isn't available.
    """
    try:
        provider = tracer_provider
        if provider is None:
            try:
                from opentelemetry.trace import get_tracer_provider as _gtp
                provider = _gtp()
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("CUA: failed to resolve global tracer provider: %s", exc)
                return
        # ``add_span_processor`` exists on the SDK ``TracerProvider``.
        add = getattr(provider, "add_span_processor", None)
        if add is None:
            return
        key = id(provider)
        if key in _framework_processor_providers:
            return
        add(_CuaFrameworkSpanProcessor())
        _framework_processor_providers.add(key)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("CUA: failed to register framework span processor: %s", exc)


class CuaInstrumentor(BaseInstrumentor):
    """Instrumentor for the CUA (Computer-Use Agent) framework."""

    _handler: Optional[ExtendedTelemetryHandler] = None
    _saved_cua_telemetry_enabled: Optional[str] = None
    _telemetry_env_overridden: bool = False

    def __init__(self) -> None:
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        # Tag the tracer provider's resource as a GenAI application so
        # downstream ARMS collectors can identify it (per gen-ai.md §应用特征).
        # Falls back to the global provider when ``tracer_provider`` is None.
        _merge_resource_attribute(tracer_provider)
        if tracer_provider is None:
            try:
                from opentelemetry.trace import get_tracer_provider as _gtp
                _merge_resource_attribute(_gtp())
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("CUA: failed to tag global tracer provider: %s", exc)

        # Register the framework-propagation span processor so LLM spans
        # (emitted by the external litellm/anthropic/openai instrumentors)
        # inherit ``gen_ai.framework=cua`` from the OTel baggage the callback
        # attaches in ``on_run_start``. Idempotent: a class-level flag avoids
        # duplicate registration on re-instrument.
        _register_framework_span_processor(tracer_provider)

        CuaInstrumentor._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        # Disable CUA's built-in OTel/PostHog telemetry to avoid double
        # instrumentation. ``CUA_TELEMETRY_ENABLED=false`` causes
        # ``OtelCallback``/``TelemetryCallback`` to no-op (they check
        # ``is_otel_enabled()`` / ``is_telemetry_enabled()`` on every hook),
        # even for agents constructed before this call.
        if not CuaInstrumentor._telemetry_env_overridden:
            CuaInstrumentor._saved_cua_telemetry_enabled = os.environ.get(
                "CUA_TELEMETRY_ENABLED"
            )
            os.environ["CUA_TELEMETRY_ENABLED"] = "false"
            CuaInstrumentor._telemetry_env_overridden = True

        # 1) Inject ArmsCuaCallback into ComputerAgent.__init__
        try:
            wrap_function_wrapper(
                module=_CUA_AGENT_MODULE,
                name=f"{_CUA_AGENT_CLASS}.__init__",
                wrapper=_wrap_agent_init,
            )
        except Exception as exc:
            logger.warning("CUA: failed to wrap ComputerAgent.__init__: %s", exc)

        # 2) Patch Sandbox lifecycle methods (TASK spans)
        tracer = get_tracer(__name__, __version__, tracer_provider=tracer_provider)
        for method, wrapper in (
            ("create", _wrap_create(tracer)),
            ("ephemeral", _wrap_ephemeral(tracer)),
            ("destroy", _wrap_destroy(tracer)),
            ("connect", _wrap_connect(tracer)),
            ("disconnect", _wrap_disconnect(tracer)),
        ):
            try:
                wrap_function_wrapper(
                    module=_CUA_SANDBOX_MODULE,
                    name=f"{_CUA_SANDBOX_CLASS}.{method}",
                    wrapper=wrapper,
                )
            except Exception as exc:
                logger.warning("CUA: failed to wrap Sandbox.%s: %s", method, exc)

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            from cua_agent.agent import ComputerAgent

            unwrap(ComputerAgent, "__init__")
            # Strip the ARMS callback from any agent that already had it injected
            for agent in getattr(self, "_instrumented_agents", []) or []:
                cb = getattr(agent, _ARMS_CALLBACK_ATTR, None)
                if cb is not None and getattr(agent, "callbacks", None):
                    try:
                        agent.callbacks.remove(cb)
                    except ValueError:
                        pass
        except Exception as exc:
            logger.debug("CUA: failed to unwrap ComputerAgent.__init__: %s", exc)

        try:
            from cua_sandbox.sandbox import Sandbox

            unwrap(Sandbox, "create")
            unwrap(Sandbox, "ephemeral")
            unwrap(Sandbox, "destroy")
            unwrap(Sandbox, "connect")
            unwrap(Sandbox, "disconnect")
        except Exception as exc:
            logger.debug("CUA: failed to unwrap Sandbox methods: %s", exc)

        CuaInstrumentor._handler = None

        # Restore the original CUA_TELEMETRY_ENABLED value so user code that
        # runs after uninstrument() sees its pre-instrumentation state.
        if CuaInstrumentor._telemetry_env_overridden:
            saved = CuaInstrumentor._saved_cua_telemetry_enabled
            if saved is None:
                os.environ.pop("CUA_TELEMETRY_ENABLED", None)
            else:
                os.environ["CUA_TELEMETRY_ENABLED"] = saved
            CuaInstrumentor._saved_cua_telemetry_enabled = None
            CuaInstrumentor._telemetry_env_overridden = False


def _wrap_agent_init(wrapped, instance, args, kwargs):
    """Wrap ComputerAgent.__init__ to attach the ARMS callback after init.

    The CUA ComputerAgent appends a number of built-in callbacks in __init__
    (OperatorNormalizerCallback, TelemetryCallback, OtelCallback, etc.).
    We let the original __init__ run, then append our callback so it sits
    last in the chain and sees the final post-processed items.
    """
    result = wrapped(*args, **kwargs)
    try:
        handler = CuaInstrumentor._handler
        if handler is None:
            return result
        # Avoid double-injection if __init__ is called twice (re-init)
        existing = getattr(instance, _ARMS_CALLBACK_ATTR, None)
        if existing is not None:
            return result
        callback = ArmsCuaCallback(handler, instance)
        if not hasattr(instance, "callbacks") or instance.callbacks is None:
            instance.callbacks = []
        instance.callbacks.append(callback)
        setattr(instance, _ARMS_CALLBACK_ATTR, callback)
    except Exception as exc:
        logger.warning("CUA: failed to inject ArmsCuaCallback: %s", exc)
    return result

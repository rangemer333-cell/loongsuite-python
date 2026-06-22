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
from typing import Any, Collection, Optional

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.cua.package import _instruments
from opentelemetry.instrumentation.cua.version import __version__
from opentelemetry.instrumentation.cua.callback import ArmsCuaCallback
from opentelemetry.instrumentation.cua.sandbox_patch import (
    _wrap_connect,
    _wrap_destroy,
    _wrap_disconnect,
    _wrap_create,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

_CUA_AGENT_MODULE = "cua_agent.agent"
_CUA_AGENT_CLASS = "ComputerAgent"
_CUA_SANDBOX_MODULE = "cua_sandbox.sandbox"
_CUA_SANDBOX_CLASS = "Sandbox"

_ARMS_CALLBACK_ATTR = "_arms_cua_callback"


class CuaInstrumentor(BaseInstrumentor):
    """Instrumentor for the CUA (Computer-Use Agent) framework."""

    _handler: Optional[ExtendedTelemetryHandler] = None

    def __init__(self) -> None:
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        CuaInstrumentor._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

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
            unwrap(Sandbox, "destroy")
            unwrap(Sandbox, "connect")
            unwrap(Sandbox, "disconnect")
        except Exception as exc:
            logger.debug("CUA: failed to unwrap Sandbox methods: %s", exc)

        CuaInstrumentor._handler = None


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

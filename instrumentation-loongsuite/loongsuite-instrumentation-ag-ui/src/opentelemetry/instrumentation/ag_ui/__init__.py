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

"""OpenTelemetry instrumentation for the AG-UI Protocol.

This instrumentation monkey-patches the AG-UI Python SDK's
``EventEncoder`` so that every SSE event flowing through ``encode()`` is
routed into the LoongSuite :class:`ExtendedTelemetryHandler`. The handler
emits four levels of spans — ENTRY, AGENT, STEP, TOOL — strictly following
the semantic conventions in ``/home/admin/semantic-conventions/arms_docs/trace/gen-ai.md``.

Patches applied at instrument time:

* ``EventEncoder.__init__`` — attach a fresh :class:`AGUISpanManager` per
  encoder instance so that each request keeps independent span state.
* ``EventEncoder.encode`` — route each typed event to the manager before
  delegating to the original encoder.
* ``RunAgentInput.model_post_init`` — store the parsed input into a
  ``ContextVar`` so that the per-encoder manager can read the request
  context (messages, tools, ``forwarded_props``).

The instrumentation is a no-op when ``AGUIConfig.enabled`` is False or when
the ``ag-ui-protocol`` package cannot be imported.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from importlib import import_module
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.ag_ui.config import AGUIConfig
from opentelemetry.instrumentation.ag_ui.package import _instruments
from opentelemetry.instrumentation.ag_ui.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
    get_extended_telemetry_handler,
)

logger = logging.getLogger(__name__)

# Module-level state populated by _instrument(); read by patched functions.
_current_agui_input: ContextVar[Any] = ContextVar(
    "agui_current_input", default=None
)
_global_handler: ExtendedTelemetryHandler | None = None
_global_config: AGUIConfig | None = None
_original_post_init: Any = None


def _patched_encoder_init(wrapped, instance, args, kwargs):
    wrapped(*args, **kwargs)
    if _global_handler is None or _global_config is None:
        return
    # Import locally so uninstrument does not need the SDK on sys.path.
    try:
        from opentelemetry.instrumentation.ag_ui.internal._span_manager import (
            AGUISpanManager,
        )
    except ImportError:  # pragma: no cover - defensive
        return
    input_data = _current_agui_input.get(None)
    instance._agui_span_manager = AGUISpanManager(
        handler=_global_handler,
        input_data=input_data,
        config=_global_config,
    )


def _patched_encoder_encode(wrapped, instance, args, kwargs):
    manager = getattr(instance, "_agui_span_manager", None)
    if manager is not None and args:
        try:
            manager.on_event(args[0])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "AG-UI span manager on_event failed: %s", exc, exc_info=True
            )
    return wrapped(*args, **kwargs)


def _patched_post_init(self, __context):
    if _original_post_init is not None:
        try:
            _original_post_init(self, __context)
        except TypeError:
            # Older pydantic signatures may differ; fall through anyway.
            pass
    _current_agui_input.set(self)


class AGUIInstrumentor(BaseInstrumentor):
    """Instrument AG-UI Protocol's ``EventEncoder`` for OpenTelemetry tracing."""

    _encoder_module = "ag_ui.encoder.encoder"
    _encoder_class = "EventEncoder"
    _input_module = "ag_ui.core.types"
    _input_class = "RunAgentInput"

    def __init__(self) -> None:
        super().__init__()
        self._handler: ExtendedTelemetryHandler | None = None
        self._config: AGUIConfig | None = None
        self._owns_handler: bool = False
        self._input_class_ref: Any = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        global _global_handler, _global_config, _original_post_init

        config = AGUIConfig.from_env()
        if not config.enabled:
            logger.info(
                "AG-UI instrumentation is disabled via "
                "OTEL_INSTRUMENTATION_AG_UI_ENABLED=false"
            )
            return

        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")
        if (
            tracer_provider is not None
            or meter_provider is not None
            or logger_provider is not None
        ):
            handler = ExtendedTelemetryHandler(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
                logger_provider=logger_provider,
            )
            self._owns_handler = True
        else:
            handler = get_extended_telemetry_handler()

        self._handler = handler
        self._config = config
        _global_handler = handler
        _global_config = config

        # Patch EventEncoder.__init__ and EventEncoder.encode.
        try:
            wrap_function_wrapper(
                module=self._encoder_module,
                name=f"{self._encoder_class}.__init__",
                wrapper=_patched_encoder_init,
            )
            wrap_function_wrapper(
                module=self._encoder_module,
                name=f"{self._encoder_class}.encode",
                wrapper=_patched_encoder_encode,
            )
        except Exception as exc:
            logger.warning(
                "AG-UI instrumentation failed to wrap EventEncoder: %s",
                exc,
                exc_info=True,
            )
            return

        # Patch RunAgentInput.model_post_init via direct assignment so that
        # the ContextVar is populated when FastAPI parses the request body.
        try:
            types_module = import_module(self._input_module)
            input_cls = getattr(types_module, self._input_class, None)
            if input_cls is None:
                logger.debug(
                    "AG-UI: %s.%s not found; skipping ContextVar patch",
                    self._input_module,
                    self._input_class,
                )
            elif getattr(input_cls, "model_post_init", None) is not None:
                self._input_class_ref = input_cls
                _original_post_init = input_cls.model_post_init
                input_cls.model_post_init = _patched_post_init  # type: ignore[assignment]
        except ImportError:
            logger.debug(
                "AG-UI types module not importable; skipping input patch",
                exc_info=True,
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        global _global_handler, _global_config, _original_post_init
        del kwargs

        try:
            encoder_module = import_module(self._encoder_module)
            encoder_cls = getattr(encoder_module, self._encoder_class, None)
            if encoder_cls is not None:
                unwrap(encoder_cls, "__init__")
                unwrap(encoder_cls, "encode")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "AG-UI instrumentation: could not unwrap EventEncoder: %s",
                exc,
            )

        if self._input_class_ref is not None and _original_post_init is not None:
            try:
                self._input_class_ref.model_post_init = _original_post_init
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "AG-UI: could not restore RunAgentInput.model_post_init: %s",
                    exc,
                )

        if self._owns_handler and self._handler is not None:
            try:
                self._handler.shutdown()
            except Exception:  # pragma: no cover - defensive
                pass

        self._handler = None
        self._config = None
        self._input_class_ref = None
        _global_handler = None
        _global_config = None
        _original_post_init = None


__all__ = [
    "__version__",
    "AGUIInstrumentor",
    "AGUIConfig",
]

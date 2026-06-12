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

"""LoongSuite instrumentation for langchain-ai DeepAgents."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Collection

from opentelemetry.instrumentation.deepagents.internal.patch import (
    instrument_create_deep_agent,
    uninstrument_create_deep_agent,
)
from opentelemetry.instrumentation.deepagents.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

__all__ = ["DeepAgentsInstrumentor"]

logger = logging.getLogger(__name__)


def _instrument_dependency(
    module_name: str,
    class_name: str,
    **kwargs: Any,
) -> None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or (
            exc.name is not None and module_name.startswith(f"{exc.name}.")
        ):
            logger.warning(
                "deepagents instrumentation requires %s; continuing without it.",
                module_name,
            )
            return
        raise

    instrumentor_type = getattr(module, class_name, None)
    if instrumentor_type is None:
        logger.warning(
            "deepagents instrumentation could not find %s.%s",
            module_name,
            class_name,
        )
        return

    instrumentor = instrumentor_type()
    if instrumentor.is_instrumented_by_opentelemetry:
        return
    instrumentor.instrument(**kwargs)


class DeepAgentsInstrumentor(BaseInstrumentor):
    """Instrument DeepAgents root graphs as LangChain agent spans."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        _instrument_dependency(
            "opentelemetry.instrumentation.langchain",
            "LangChainInstrumentor",
            **kwargs,
        )
        _instrument_dependency(
            "opentelemetry.instrumentation.langgraph",
            "LangGraphInstrumentor",
            **kwargs,
        )
        instrument_create_deep_agent()

    def _uninstrument(self, **kwargs: Any) -> None:
        uninstrument_create_deep_agent()

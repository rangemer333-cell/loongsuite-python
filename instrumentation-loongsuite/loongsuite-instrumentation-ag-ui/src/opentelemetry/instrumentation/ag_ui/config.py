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

"""Configuration for AG-UI instrumentation."""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_MAX_CONTENT_LENGTH = 65536


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _content_capture_enabled() -> bool:
    """Mirror the util-genai content capturing gate.

    The util-genai layer only writes content-typed attributes
    (``gen_ai.input.messages``, ``gen_ai.tool.call.arguments``, ...) when
    both experimental mode is opted-in and the capture mode is SPAN-only.
    This helper keeps AGUIConfig in sync with that gate so users do not need
    to set a separate flag for the AG-UI instrumentation.
    """
    try:
        from opentelemetry.util.genai.utils import (
            get_content_capturing_mode,
            is_experimental_mode,
        )
        from opentelemetry.util.genai.types import ContentCapturingMode
    except ImportError:
        return False
    try:
        if not is_experimental_mode():
            return False
        mode = get_content_capturing_mode()
        return mode in (ContentCapturingMode.SPAN_ONLY, ContentCapturingMode.SPAN_AND_EVENT)
    except Exception:
        return False


@dataclass
class AGUIConfig:
    """Runtime configuration for the AG-UI instrumentation.

    Attributes:
        capture_content: when True, gen_ai.input.messages / gen_ai.output.messages
            / gen_ai.tool.call.arguments / gen_ai.tool.call.result are captured on
            spans. Derived from the util-genai content capturing mode so that the
            AG-UI instrumentation stays in sync with the rest of the LoongSuite
            GenAI instrumentations (requires
            ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` and
            ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`` or
            ``SPAN_AND_EVENT``).
        max_content_length: maximum number of bytes serialized into a single
            content-typed span attribute. Content exceeding this limit is
            truncated and flagged.
        enabled: master switch; when False the instrumentor is a no-op.
    """

    capture_content: bool = False
    max_content_length: int = _DEFAULT_MAX_CONTENT_LENGTH
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "AGUIConfig":
        enabled = _env_truthy("OTEL_INSTRUMENTATION_AG_UI_ENABLED", True)
        max_len = _DEFAULT_MAX_CONTENT_LENGTH
        raw_len = os.environ.get(
            "OTEL_INSTRUMENTATION_AG_UI_MAX_CONTENT_LENGTH"
        )
        if raw_len:
            try:
                max_len = max(1, int(raw_len))
            except ValueError:
                pass
        return cls(
            capture_content=_content_capture_enabled(),
            max_content_length=max_len,
            enabled=enabled,
        )

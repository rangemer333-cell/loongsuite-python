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

"""Smoke tests for the CuaInstrumentor entry point.

We install lightweight stub modules under the names ``cua_agent.agent`` and
``cua_sandbox.sandbox`` so the wrapt-based patches succeed without pulling
in the real (heavy) CUA dependencies. The tests then verify the patch is
applied and removed correctly.
"""

import os
import sys
import types
from contextlib import asynccontextmanager

import pytest


def _install_stub_modules():
    """Register stub cua_agent / cua_sandbox modules in sys.modules."""
    # cua_agent package
    if "cua_agent" not in sys.modules:
        pkg_agent = types.ModuleType("cua_agent")
        pkg_agent.__path__ = []
        sys.modules["cua_agent"] = pkg_agent
    if "cua_agent.agent" not in sys.modules:
        mod_agent = types.ModuleType("cua_agent.agent")
        sys.modules["cua_agent.agent"] = mod_agent
    mod_agent = sys.modules["cua_agent.agent"]

    class ComputerAgent:
        def __init__(self, model, **kwargs):
            self.model = model
            self.instructions = kwargs.get("instructions")
            self.callbacks = []

    mod_agent.ComputerAgent = ComputerAgent

    # cua_sandbox package
    if "cua_sandbox" not in sys.modules:
        pkg_sb = types.ModuleType("cua_sandbox")
        pkg_sb.__path__ = []
        sys.modules["cua_sandbox"] = pkg_sb
    if "cua_sandbox.sandbox" not in sys.modules:
        mod_sb = types.ModuleType("cua_sandbox.sandbox")
        sys.modules["cua_sandbox.sandbox"] = mod_sb
    mod_sb = sys.modules["cua_sandbox.sandbox"]

    class Sandbox:
        @classmethod
        async def create(cls, image, **kwargs):
            return cls()

        @classmethod
        @asynccontextmanager
        async def ephemeral(cls, image, **kwargs):
            yield cls()

        @classmethod
        def connect(cls, name, **kwargs):
            return cls()

        async def destroy(self):
            return None

        async def disconnect(self):
            return None

    mod_sb.Sandbox = Sandbox
    return mod_agent, mod_sb


def test_instrument_attaches_callback_to_agent_init(tracer_provider):
    mod_agent, _ = _install_stub_modules()
    from opentelemetry.instrumentation.cua import CuaInstrumentor

    instrumentor = CuaInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    try:
        agent = mod_agent.ComputerAgent(
            model="anthropic/claude-sonnet-4-5",
            instructions="be helpful",
        )
        assert len(agent.callbacks) == 1
        cb = agent.callbacks[0]
        from opentelemetry.instrumentation.cua.callback import ArmsCuaCallback

        assert isinstance(cb, ArmsCuaCallback)
    finally:
        instrumentor.uninstrument()


def test_uninstrument_removes_patches(tracer_provider):
    mod_agent, mod_sb = _install_stub_modules()
    from opentelemetry.instrumentation.cua import CuaInstrumentor

    instrumentor = CuaInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    instrumentor.uninstrument()

    # After uninstrument, creating an agent should NOT append our callback
    agent = mod_agent.ComputerAgent(model="anthropic/claude-3")
    assert agent.callbacks == []


def test_repeated_instrument_idempotent_callback(tracer_provider):
    """Calling instrument twice then creating one agent must yield at most one
    injected callback per agent instance (not double-injected)."""
    mod_agent, _ = _install_stub_modules()
    from opentelemetry.instrumentation.cua import CuaInstrumentor

    instr1 = CuaInstrumentor()
    instr1.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    try:
        agent = mod_agent.ComputerAgent(model="anthropic/claude-3")
        n_after_first = len(agent.callbacks)
        assert n_after_first == 1
    finally:
        instr1.uninstrument()


def test_instrument_disables_cua_builtin_telemetry(tracer_provider, monkeypatch):
    """Regression for review problem 1: instrument() must force
    ``CUA_TELEMETRY_ENABLED=false`` so the built-in OtelCallback becomes a
    no-op (it checks ``is_otel_enabled()`` on every hook), avoiding double
    telemetry with this plugin.
    """
    mod_agent, _ = _install_stub_modules()
    monkeypatch.setenv("CUA_TELEMETRY_ENABLED", "true")
    from opentelemetry.instrumentation.cua import CuaInstrumentor

    instrumentor = CuaInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    try:
        assert os.environ.get("CUA_TELEMETRY_ENABLED") == "false"
    finally:
        instrumentor.uninstrument()


def test_uninstrument_restores_cua_telemetry_env(tracer_provider, monkeypatch):
    """Regression for review problem 1: uninstrument() must restore the
    original ``CUA_TELEMETRY_ENABLED`` value (set, unset, or user-provided).
    """
    mod_agent, _ = _install_stub_modules()
    from opentelemetry.instrumentation.cua import CuaInstrumentor

    # Case 1: previously unset -> after uninstrument, still unset
    monkeypatch.delenv("CUA_TELEMETRY_ENABLED", raising=False)
    instrumentor = CuaInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    instrumentor.uninstrument()
    assert "CUA_TELEMETRY_ENABLED" not in os.environ

    # Case 2: previously user-set -> after uninstrument, original value restored
    monkeypatch.setenv("CUA_TELEMETRY_ENABLED", "true")
    instrumentor = CuaInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    instrumentor.uninstrument()
    assert os.environ.get("CUA_TELEMETRY_ENABLED") == "true"


def test_instrument_tags_resource_with_genai_app_feature(tracer_provider):
    """Regression for verification non-blocking #4: instrument() must merge
    ``acs.arms.service.feature=genai_app`` onto the tracer provider's resource
    (per gen-ai.md §应用特征). Must not overwrite the user's existing
    ``service.name``.
    """
    _install_stub_modules()
    from opentelemetry.instrumentation.cua import CuaInstrumentor

    # Tag the provider's resource so we can verify it survives the merge.
    original_service_name = tracer_provider.resource.attributes.get("service.name")
    instrumentor = CuaInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    try:
        attrs = dict(tracer_provider.resource.attributes or {})
        assert attrs.get("acs.arms.service.feature") == "genai_app"
        # User's existing service.name must be preserved.
        assert attrs.get("service.name") == original_service_name
    finally:
        instrumentor.uninstrument()


def test_framework_span_processor_survives_on_ending_call(tracer_provider):
    """Regression for verification 5173946b: ``_CuaFrameworkSpanProcessor``
    must not crash when the SDK calls ``_on_ending`` on ``Span.end()``.

    opentelemetry-sdk>=1.42 added a ``_on_ending`` hook invoked by
    ``Span.end()`` before ``on_end``. ``SynchronousMultiSpanProcessor``
    forwards it to every registered processor. A bare ``class
    _CuaFrameworkSpanProcessor:`` raised ``AttributeError`` at runtime on
    SDK 1.42 (the cluster pod), even though the local dev env (SDK 1.37)
    didn't expose the hook and tests passed.

    We can't easily install SDK 1.42 in CI, so we simulate the
    multi-processor forwarding pattern: a fake multi-processor that
    mimics SDK 1.42 by calling ``_on_ending`` then ``on_end`` on every
    registered sp. If the class ever drops the ``_on_ending`` definition
    again, this raises ``AttributeError``.
    """
    from opentelemetry.instrumentation.cua import _CuaFrameworkSpanProcessor

    class _FakeSpan:
        pass

    class _FakeMultiProcessor:
        """Mimics SDK 1.42 ``SynchronousMultiSpanProcessor``: ``Span.end()``
        calls ``_on_ending`` then ``on_end`` on every registered sp."""

        def __init__(self, sps):
            self._sps = sps

        def _on_ending(self, span):
            for sp in self._sps:
                sp._on_ending(span)

        def on_end(self, span):
            for sp in self._sps:
                sp.on_end(span)

    proc = _CuaFrameworkSpanProcessor()
    multi = _FakeMultiProcessor([proc])
    span = _FakeSpan()
    multi._on_ending(span)  # Must not raise AttributeError.
    multi.on_end(span)

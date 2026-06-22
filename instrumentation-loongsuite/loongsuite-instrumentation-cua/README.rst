LoongSuite Instrumentation for CUA (Computer-Use Agent)
========================================================

This library provides automatic instrumentation for the `CUA framework
<https://github.com/trycua/cua>`_, adding OpenTelemetry tracing and metrics
for ComputerAgent runs and Sandbox lifecycle operations.

.. note::
   This package is currently in development and must be installed from source.

Installation
------------

::

    pip install opentelemetry-distro opentelemetry-exporter-otlp
    opentelemetry-bootstrap -a install

    pip install cua-agent cua-sandbox

    # Install this instrumentation
    pip install ./instrumentation-loongsuite/loongsuite-instrumentation-cua

    # Required shared utility
    pip install ./util/opentelemetry-util-genai

Usage
-----

Auto-instrumentation
~~~~~~~~~~~~~~~~~~~~

Use the ``opentelemetry-instrument`` wrapper::

    opentelemetry-instrument \
        --traces_exporter console \
        --metrics_exporter console \
        python your_script.py

Or instrument programmatically::

    from opentelemetry.instrumentation.cua import CuaInstrumentor

    CuaInstrumentor().instrument()

Captured spans
--------------

The instrumentation emits the following spans following the ARMS GenAI
semantic conventions:

- ``enter_ai_application_system`` (ENTRY) — top-level span around an
  ``agent.run`` invocation. Carries ``gen_ai.session.id`` and
  ``gen_ai.user.id``.
- ``invoke_agent {model}`` (AGENT) — child of ENTRY; carries agent name,
  description, id, conversation id, and token usage.
- ``react step`` (STEP) — child of AGENT; one per LLM round, carries
  ``gen_ai.react.round`` and ``gen_ai.react.finish_reason``.
- ``execute_tool {action_type}`` (TOOL) — child of STEP; covers both
  computer-use actions (click, type, scroll, screenshot, ...) and custom
  function calls. Carries ``gen_ai.tool.name``, ``gen_ai.tool.type``,
  ``gen_ai.tool.call.id``, plus ``cua.action.*`` extension attributes.
- ``run_task sandbox.{operation}`` (TASK) — independent spans around
  ``Sandbox.create / destroy / connect / disconnect``. Carry
  ``cua.sandbox.*`` attributes (name, image, runtime, region, ...).

LLM spans are produced by the existing LiteLLM/OpenAI/Anthropic
instrumentations — CUA drives LLM calls through ``litellm`` so we don't
duplicate them here.

Configuration
-------------

- ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``: controls whether
  message bodies, tool arguments, and tool results are captured. Defaults
  to ``NO_CONTENT`` (no capture). Set to ``SPAN_ONLY`` or ``SPAN_AND_EVENT``
  to enable.
- ``OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH``: maximum length
  (in characters) of captured message content. Default 8192.

License
-------

Apache-2.0

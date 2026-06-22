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
- On ``instrument()``, the instrumentor forces
  ``CUA_TELEMETRY_ENABLED=false`` to disable CUA's built-in
  ``OtelCallback`` / ``TelemetryCallback`` (which would otherwise emit a
  second, ARMS-unaligned trace to ``otel.cua.ai``). The original value is
  restored on ``uninstrument()``.

Known limitations
-----------------

The following are tracked for the next iteration and do not block deployment:

- **STEP span close timing**: CUA executes tool calls *after* ``on_responses``
  returns, so STEP is closed on the next ``on_llm_start`` (or ``on_run_end``)
  rather than in ``on_responses`` to preserve the STEP → TOOL parent/child
  link. See ``execute.md`` §4.3 for the deviation note.
- **``uninstrument()`` does not strip ``ArmsCuaCallback`` from existing
  ``ComputerAgent`` instances**. The class-level ``_handler`` is cleared on
  uninstrument, so subsequent callback invocations become no-ops, but the
  callback object remains in ``agent.callbacks``. A future iteration will
  track instrumented instances and remove the callback explicitly.
- **``gen_ai.tool.type = "computer_use"``** is an ARMS extension of the
  semantic-convention enum (``function`` / ``extension`` / ``datastore``).
  Consumers must recognise this value for CUA computer-use actions.
- **``Sandbox.connect`` is patched at the sync entry** rather than the
  recommended ``_connect`` async method (``execute.md`` §4.7). The
  connection-duration therefore includes ``_ConnectResult`` construction
  plus lazy await. Switch to patching ``_connect`` if sub-method
  granularity is needed.
- **``_fail_open_step_span`` in ``callback.py`` is currently unused**
  (dead code); the single ``_close_open_step_span`` path is used for both
  normal and fail-close. Will be removed or wired up in a future iteration.

License
-------

Apache-2.0

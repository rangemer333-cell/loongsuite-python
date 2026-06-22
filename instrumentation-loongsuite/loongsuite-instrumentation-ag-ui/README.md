# loongsuite-instrumentation-ag-ui

OpenTelemetry instrumentation for the
[AG-UI Protocol](https://github.com/ag-ui-protocol/ag-ui) Python SDK
(`ag-ui-protocol`).

This instrumentation monkey-patches the SDK's `EventEncoder.encode()` so that
every SSE event flowing through an AG-UI integration (LangGraph, CrewAI,
Strands, Claude SDK, ...) is routed into the LoongSuite
`ExtendedTelemetryHandler`. Four levels of spans are emitted — ENTRY →
AGENT → STEP → TOOL — strictly following the semantic conventions defined
in `/home/admin/semantic-conventions/arms_docs/trace/gen-ai.md`.

## Span lifecycle

| AG-UI event | Span created/closed |
|-------------|--------------------|
| `RUN_STARTED`   | open ENTRY + AGENT |
| `STEP_STARTED`   | open STEP |
| `TOOL_CALL_START` | open TOOL |
| `TOOL_CALL_ARGS`  | accumulate arguments |
| `TOOL_CALL_END` / `TOOL_CALL_RESULT` | close TOOL |
| `STEP_FINISHED`  | close STEP |
| `TEXT_MESSAGE_CONTENT` | record TTFT on ENTRY/AGENT (no span) |
| `MESSAGES_SNAPSHOT` | cache last snapshot for `gen_ai.output.messages` |
| `RUN_FINISHED`  | close AGENT + ENTRY |
| `RUN_ERROR`     | fail AGENT + ENTRY and any open STEP/TOOL |

LLM spans are **not** emitted here — AG-UI events do not carry model name,
provider, or token usage. Those are captured by the underlying LLM SDK
instrumentation (openai/litellm/dashscope) via OTel context propagation,
making them children of the STEP/AGENT spans.

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `OTEL_INSTRUMENTATION_AG_UI_ENABLED` | `true` | Master switch. |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false` | When truthy,
  captures `gen_ai.input.messages`, `gen_ai.output.messages`,
  `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` (truncated to
  `OTEL_INSTRUMENTATION_AG_UI_MAX_CONTENT_LENGTH`). |
| `OTEL_INSTRUMENTATION_AG_UI_MAX_CONTENT_LENGTH` | `65536` | Maximum length
  of any captured content-typed span attribute. |

## Installation

```sh
pip install -e ./util/opentelemetry-util-genai
pip install -e ./instrumentation-loongsuite/loongsuite-instrumentation-ag-ui
```

## Usage

```python
from opentelemetry.instrumentation.ag_ui import AGUIInstrumentor

AGUIInstrumentor().instrument()
```

Or via the `opentelemetry-instrument` CLI entry point:

```
opentelemetry-instrument --instruments ag_ui python your_app.py
```

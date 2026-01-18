# Review: Distributed TaskContext (Follow-up After Fixes)

## Findings

- High: Distributed elicitation/sampling still does not emit `related-task` metadata. The forwarder calls `ServerSession.elicit/create_message` with `related_request_id`, which only produces `related-request` metadata; embedded mode uses `_build_*` with `related_task_id`, so distributed clients lose task correlation. `src/fastmcp/server/tasks/forwarder.py:200` `src/fastmcp/server/tasks/forwarder.py:303`
- Medium: `model_preferences` is still dropped because `TaskContext.sample()` does not pass it into `_sample_distributed`. `src/fastmcp/server/dependencies.py:1241`
- Medium: Forwarder content parsing only supports text and image; audio (and other supported content types) will be mis-parsed or rejected. `src/fastmcp/server/tasks/forwarder.py:264`
- Low: Test coverage still misses critical distributed behaviors (related-task metadata, input_required transitions, model_preferences pass-through, non-text content). `tests/server/tasks/test_distributed_task_context.py:1`

## Proposed Fixes (Explanation + Code Examples)

### 1) Preserve `related-task` metadata in distributed mode

Problem: `ServerSession.elicit/create_message` only attaches `related_request_id`,
which is not the same as task metadata. Embedded mode uses `_build_*` helpers to
emit `io.modelcontextprotocol/related-task`. Distributed forwarder should mirror
that behavior.

Fix: Use the same request-building path as embedded `TaskContext` to include
`related_task_id`. This matches existing behavior and avoids metadata drift.

```python
# src/fastmcp/server/tasks/forwarder.py
import anyio
import mcp.shared.message
import mcp.shared.exceptions
import mcp.types

async def _send_elicit_with_related_task(
    session: ServerSession,
    message: str,
    schema: dict[str, Any],
    related_task_id: str,
) -> mcp.types.ElicitResult:
    request = session._build_elicit_form_request(  # pyright: ignore[reportPrivateUsage]
        message=message,
        requestedSchema=schema,
        related_task_id=related_task_id,
    )
    response_stream, response_stream_reader = anyio.create_memory_object_stream[
        mcp.types.JSONRPCResponse | mcp.types.JSONRPCError
    ](1)
    request_id = request.id
    session._response_streams[request_id] = response_stream  # pyright: ignore[reportPrivateUsage]
    try:
        await session._write_stream.send(  # pyright: ignore[reportPrivateUsage]
            mcp.shared.message.SessionMessage(message=mcp.types.JSONRPCMessage(request))
        )
        response_or_error = await response_stream_reader.receive()
        if isinstance(response_or_error, mcp.types.JSONRPCError):
            raise mcp.shared.exceptions.McpError(response_or_error.error)
        return mcp.types.ElicitResult.model_validate(response_or_error.result)
    finally:
        session._response_streams.pop(request_id, None)  # pyright: ignore[reportPrivateUsage]
        await response_stream.aclose()
        await response_stream_reader.aclose()
```

```python
# src/fastmcp/server/tasks/forwarder.py
async def _send_sample_with_related_task(
    session: ServerSession,
    messages: list[mcp.types.SamplingMessage],
    related_task_id: str,
    *,
    max_tokens: int,
    system_prompt: str | None,
    temperature: float | None,
    model_preferences: mcp.types.ModelPreferences | None,
) -> mcp.types.CreateMessageResult:
    request = session._build_create_message_request(  # pyright: ignore[reportPrivateUsage]
        messages=messages,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        temperature=temperature,
        model_preferences=model_preferences,
        related_task_id=related_task_id,
    )
    response_stream, response_stream_reader = anyio.create_memory_object_stream[
        mcp.types.JSONRPCResponse | mcp.types.JSONRPCError
    ](1)
    request_id = request.id
    session._response_streams[request_id] = response_stream  # pyright: ignore[reportPrivateUsage]
    try:
        await session._write_stream.send(  # pyright: ignore[reportPrivateUsage]
            mcp.shared.message.SessionMessage(message=mcp.types.JSONRPCMessage(request))
        )
        response_or_error = await response_stream_reader.receive()
        if isinstance(response_or_error, mcp.types.JSONRPCError):
            raise mcp.shared.exceptions.McpError(response_or_error.error)
        return mcp.types.CreateMessageResult.model_validate(response_or_error.result)
    finally:
        session._response_streams.pop(request_id, None)  # pyright: ignore[reportPrivateUsage]
        await response_stream.aclose()
        await response_stream_reader.aclose()
```

Then call these helpers inside `_handle_elicit_request` and `_handle_sample_request`
instead of `session.elicit/create_message`. This keeps distributed mode aligned
with embedded mode semantics.

### 2) Pass `model_preferences` through the distributed path

Problem: `TaskContext.sample()` currently calls `_sample_distributed(...)` without
forwarding `model_preferences`, so the distributed path ignores caller intent.

Fix: Pass the argument through and keep the existing serialization in
`_sample_distributed`.

```python
# src/fastmcp/server/dependencies.py
if self._distributed:
    return await self._sample_distributed(
        messages,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        temperature=temperature,
        model_preferences=model_preferences,
    )
```

### 3) Use the sampling content union for parsing

Problem: Forwarder parsing only recognizes text/image. The MCP sampling content
union also includes audio, and future types may be added.

Fix: Use a `TypeAdapter` over `SamplingContent` so any supported content is
validated and preserved.

```python
# src/fastmcp/server/tasks/forwarder.py
from pydantic import TypeAdapter
from mcp.types import SamplingContent, SamplingMessage, TextContent

content_adapter = TypeAdapter(SamplingContent)

for m in request_data["messages"]:
    role = m.get("role", "user")
    raw_content = m.get("content", {})
    if isinstance(raw_content, str):
        content = TextContent(type="text", text=raw_content)
    else:
        content = content_adapter.validate_python(raw_content)
    messages.append(SamplingMessage(role=role, content=content))
```

## Suggested Tests

- Forwarder emits `related-task` metadata (assert in the built JSONRPC request).
- Distributed sampling honors `model_preferences` (serialize, forward, assert).
- Parsing of audio content in distributed sampling (round-trip).
- Input-required status transitions in distributed elicit/sample (notification observed).

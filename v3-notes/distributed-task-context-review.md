# Review: Distributed TaskContext (feature/distributed-task-context)

## Findings

- High: `TaskContext` flips to distributed mode solely on missing session and ignores `FASTMCP_DISTRIBUTED_WORKERS`, so internal or disconnected sessions will attempt Redis with no forwarder and hang until timeout. `src/fastmcp/server/dependencies.py:931` `src/fastmcp/server/tasks/handlers.py:104`
- High: Forwarder bypasses SEP-1686 status semantics and related-task metadata; it calls `session.elicit`/`create_message` without `related_task_id` and does not send `input_required`/`working` notifications. `src/fastmcp/server/tasks/forwarder.py:182` `src/fastmcp/server/tasks/forwarder.py:225`
- Medium: `model_preferences` is dropped in distributed sampling (not serialized, not forwarded to `create_message`). `src/fastmcp/server/dependencies.py:1218` `src/fastmcp/server/tasks/redis_proxy.py:161` `src/fastmcp/server/tasks/forwarder.py:240`
- Medium: Forwarder hardcodes sampling content parsing to `TextContent`, so non-text content will be lost or rejected. `src/fastmcp/server/tasks/forwarder.py:225`
- Medium: `stop_forwarders_for_session` is not wired to any session lifecycle hook, so forwarders can leak after session disconnects. `src/fastmcp/server/tasks/forwarder.py:315`
- Low: Tests only cover flags/datatypes and do not exercise Redis proxy/forwarder round-trips or status transitions. `tests/server/tasks/test_distributed_task_context.py:1`

## Proposed Fixes (Explanation + Code Examples)

### 1) Gate distributed mode and fail fast for unsupported sessions

Problem: TaskContext infers distributed mode solely by missing session, even when
the feature flag is off or when the session is "internal". This leads to Redis
calls that cannot succeed.

Fix: Track the feature flag and validate before distributed calls. If disabled,
raise a clear error early. Optionally guard internal sessions explicitly.

```python
# src/fastmcp/server/dependencies.py
from fastmcp.server.tasks.forwarder import is_distributed_mode_enabled

class TaskContext:
    __slots__ = ("_distributed", "_distributed_enabled", "_session_id", "_task_id")

    def __init__(self, task_id: str, session_id: str) -> None:
        self._task_id = task_id
        self._session_id = session_id
        self._distributed_enabled = is_distributed_mode_enabled()
        self._distributed = get_task_session(session_id) is None

    def _ensure_distributed_enabled(self) -> None:
        if not self._distributed_enabled:
            raise RuntimeError(
                "Distributed TaskContext is disabled. "
                "Set FASTMCP_DISTRIBUTED_WORKERS=1 to enable."
            )
        if self._session_id == "internal":
            raise RuntimeError(
                "TaskContext.elicit/sample requires an MCP session; "
                "internal sessions do not support distributed mode."
            )

    async def _elicit_distributed(self, message: str, response_type: type | None = None) -> Any:
        self._ensure_distributed_enabled()
        ...

    async def _sample_distributed(self, messages: list[Any], *, max_tokens: int = 512,
                                  system_prompt: str | None = None, temperature: float | None = None) -> Any:
        self._ensure_distributed_enabled()
        ...
```

### 2) Preserve SEP-1686 status transitions and related-task metadata in the forwarder

Problem: Distributed mode currently skips `input_required` notifications and
`related_task_id` metadata, so clients lose task correlation and status changes.

Fix: Mirror embedded behavior by sending status transitions and adding
`related_task_id` when forwarding.

```python
# src/fastmcp/server/tasks/forwarder.py
from fastmcp.server.tasks.subscriptions import send_input_required_notification

async def _handle_elicit_request(self, data: bytes | str) -> None:
    ...
    try:
        await send_input_required_notification(
            session=self.session,
            task_id=self.task_id,
            session_id=self.session_id,
            docket=self.docket,
            status="input_required",
        )

        result = await self.session.elicit(
            message=request_data["message"],
            requestedSchema=request_data.get("schema", {}),
            related_task_id=self.task_id,
        )
        ...
    finally:
        with suppress(Exception):
            await send_input_required_notification(
                session=self.session,
                task_id=self.task_id,
                session_id=self.session_id,
                docket=self.docket,
                status="working",
            )
```

```python
# src/fastmcp/server/tasks/forwarder.py
async def _handle_sample_request(self, data: bytes | str) -> None:
    ...
    try:
        await send_input_required_notification(
            session=self.session,
            task_id=self.task_id,
            session_id=self.session_id,
            docket=self.docket,
            status="input_required",
        )

        result = await self.session.create_message(
            messages=messages,
            max_tokens=request_data.get("max_tokens", 512),
            system_prompt=request_data.get("system_prompt"),
            temperature=request_data.get("temperature"),
            model_preferences=request_data.get("model_preferences"),
            related_task_id=self.task_id,
        )
        ...
    finally:
        with suppress(Exception):
            await send_input_required_notification(
                session=self.session,
                task_id=self.task_id,
                session_id=self.session_id,
                docket=self.docket,
                status="working",
            )
```

### 3) Round-trip model_preferences through Redis

Problem: `model_preferences` is ignored on the distributed path, which can change
sampling behavior vs embedded mode.

Fix: Serialize it in the worker, pass it through Redis, and forward it to
`create_message`.

```python
# src/fastmcp/server/dependencies.py
async def _sample_distributed(..., model_preferences: Any | None = None) -> Any:
    ...
    result_data = await send_sample_via_redis(
        ...,
        model_preferences=(
            model_preferences.model_dump(mode="json")
            if hasattr(model_preferences, "model_dump")
            else model_preferences
        ),
    )
```

```python
# src/fastmcp/server/tasks/redis_proxy.py
async def send_sample_via_redis(..., model_preferences: dict[str, Any] | None = None, ...) -> dict[str, Any]:
    request_payload = json.dumps(
        {
            ...,
            "model_preferences": model_preferences,
        }
    )
```

```python
# src/fastmcp/server/tasks/forwarder.py
result = await self.session.create_message(
    ...,
    model_preferences=request_data.get("model_preferences"),
)
```

### 4) Support non-text sampling content in distributed mode

Problem: The forwarder coerces everything into `TextContent`, losing image/audio
content and causing validation errors.

Fix: Validate content using the MCP content union so non-text types survive.

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

### 5) Hook forwarder cleanup into session lifecycle

Problem: Forwarders can outlive sessions without a disconnect hook.

Fix: Register cleanup on session exit (mirrors the pattern in
`src/fastmcp/server/providers/proxy.py`).

```python
# src/fastmcp/server/tasks/handlers.py
from fastmcp.server.tasks.forwarder import stop_forwarders_for_session

if getattr(ctx.session, "_exit_stack", None) is not None:
    ctx.session._exit_stack.push_async_callback(
        stop_forwarders_for_session, session_id
    )
```

## Suggested Follow-Up Tests

- End-to-end Redis round-trip for elicit/sample with `input_required` transitions.
- Distributed sampling with `model_preferences` to ensure it is preserved.
- Sampling with non-text content (image/audio) to validate content parsing.

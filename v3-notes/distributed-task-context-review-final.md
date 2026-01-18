# Review: Distributed TaskContext (Post‑Followup Implementation)

## Findings

- Medium: Distributed sampling cannot serialize multi‑block content because `_sample_distributed` assumes `m.content` is a single model with `model_dump`; if `SamplingMessage.content` is a list, this raises and breaks sampling. `src/fastmcp/server/dependencies.py:1355`
- Medium: Forwarder parses sampling content using `SamplingContent` (text/image/audio only). This rejects valid `SamplingMessage.content` variants like tool use/result or lists of blocks, so distributed sampling fails for tool‑enabled or multi‑part messages. `src/fastmcp/server/tasks/forwarder.py:290`
- Low: `is_distributed` can be True even when distributed mode is disabled; callers may treat it as a capability and then hit runtime errors in `_ensure_distributed_enabled`. `src/fastmcp/server/dependencies.py:951`
- Low: Tests still do not cover Redis round‑trip paths or distributed status/metadata behavior. `tests/server/tasks/test_distributed_task_context.py:1`

## Proposed Fixes (Explanation + Code Examples)

### 1) Serialize multi‑block content in distributed sampling

Problem: `SamplingMessage.content` can be a list of content blocks, but the
distributed path always calls `m.content.model_dump(...)`, which fails for lists.

Fix: Normalize content to JSON whether it’s a single block or a list.

```python
# src/fastmcp/server/dependencies.py
def _dump_sampling_content(content: Any) -> Any:
    if isinstance(content, list):
        return [block.model_dump(mode="json") for block in content]
    return content.model_dump(mode="json")

messages_data = [
    {"role": m.role, "content": _dump_sampling_content(m.content)}
    for m in sampling_messages
]
```

### 2) Parse full SamplingMessage content in the forwarder

Problem: Forwarder uses `SamplingContent`, which excludes tool use/result and
list‑of‑blocks. This rejects valid MCP sampling payloads.

Fix: Validate against the same union that `SamplingMessage.content` allows.

```python
# src/fastmcp/server/tasks/forwarder.py
from pydantic import TypeAdapter
from mcp.types import SamplingMessage, TextContent

content_adapter = TypeAdapter(SamplingMessage.model_fields["content"].annotation)

for m in request_data["messages"]:
    role = m.get("role", "user")
    raw_content = m.get("content", {})
    if isinstance(raw_content, str):
        content = TextContent(type="text", text=raw_content)
    else:
        content = content_adapter.validate_python(raw_content)
    messages.append(SamplingMessage(role=role, content=content))
```

### 3) Align `is_distributed` with feature flag semantics

Problem: `is_distributed` reflects “session missing” rather than “distributed
mode is enabled,” which can mislead callers.

Fix: Make the property report actual distributed capability, or add a second
property to distinguish “session missing” vs “enabled”.

```python
# src/fastmcp/server/dependencies.py
@property
def is_distributed(self) -> bool:
    return self._distributed and self._distributed_enabled

@property
def session_missing(self) -> bool:
    return self._distributed
```

## Suggested Tests

- Round‑trip distributed sampling with list‑of‑blocks content (assert success).
- Forwarder accepts tool‑use/tool‑result content in messages.
- `is_distributed` reflects feature flag state (enabled vs disabled).

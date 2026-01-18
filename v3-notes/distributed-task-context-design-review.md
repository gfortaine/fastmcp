# Distributed TaskContext Design Review

## Findings

- High: Timeout handling will not fire because `pubsub.listen()` blocks until a message arrives; the deadline check only runs after a message, so no-response or disconnect cases can hang indefinitely. `v3-notes/distributed-task-context-design.md:193` `v3-notes/distributed-task-context-design.md:276`
- High: Channel routing uses substring checks (`"elicit" in channel` / `"sample" in channel`), which can misroute if `session_id` or `task_id` contains those substrings; parse segments or match exact channels. `v3-notes/distributed-task-context-design.md:392`
- Medium: Sampling response schema is inconsistent: the message format shows top-level `role/content/model`, but the worker expects `"result"` and the forwarder publishes `"result"`. This will break interop if implemented literally. `v3-notes/distributed-task-context-design.md:78` `v3-notes/distributed-task-context-design.md:521`
- Medium: Forwarder processes requests serially per task; `await self._session.elicit()` blocks the pubsub loop, which conflicts with the proposed concurrent-elicit test and can trigger timeouts/head-of-line blocking. `v3-notes/distributed-task-context-design.md:435` `v3-notes/distributed-task-context-design.md:1042`
- Low: Redis Pub/Sub is lossy; restarts or transient disconnects drop requests without any delivery guarantee beyond a timeout, so reliability expectations should be called out or Streams considered. `v3-notes/distributed-task-context-design.md:9`

## Open Questions

- Should `FASTMCP_DISTRIBUTED_WORKERS` gate `start_forwarder()` and the distributed path, since the flag is defined but the forwarder is started unconditionally in the snippet? `v3-notes/distributed-task-context-design.md:713` `v3-notes/distributed-task-context-design.md:822`
- Is `input_required` the intended status for sampling? If not, the UI/client semantics may be misleading. `v3-notes/distributed-task-context-design.md:492`

## Change Summary

- Adds a design proposal for Redis-backed distributed `TaskContext`, including proxy/forwarder sketches and a test plan in `v3-notes/distributed-task-context-design.md`.

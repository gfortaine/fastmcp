"""Forwarder bridge between Redis Pub/Sub and ServerSession.

This module runs on the FastMCP server side and bridges distributed worker
requests (arriving via Redis Pub/Sub) to the actual ServerSession. It
subscribes to request channels and forwards elicitation/sampling requests
to the session, then publishes responses back via Redis.

The forwarder is started automatically when:
1. A task is submitted
2. FASTMCP_DISTRIBUTED_WORKERS=1 is set

Each forwarder handles one (session_id, task_id) pair.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastmcp.utilities.logging import get_logger

if TYPE_CHECKING:
    from docket import Docket
    from mcp.server.session import ServerSession

logger = get_logger(__name__)


def is_distributed_mode_enabled() -> bool:
    """Check if distributed workers feature is enabled."""
    return os.getenv("FASTMCP_DISTRIBUTED_WORKERS", "").lower() in ("1", "true", "yes")


@dataclass
class ForwarderKey:
    """Unique identifier for a forwarder instance."""

    session_id: str
    task_id: str

    def __hash__(self) -> int:
        return hash((self.session_id, self.task_id))


@dataclass
class ElicitForwarder:
    """Bridges elicitation requests from Redis to ServerSession.

    This forwarder:
    1. Subscribes to the elicitation request channel for its task
    2. When a request arrives, spawns a handler task (non-blocking)
    3. Calls session.elicit() with the request parameters
    4. Publishes the result to the response channel

    Critical implementation notes:
    - Uses exact channel matching (stores channel var, compares equality)
    - Spawns asyncio.create_task() for each request to handle concurrency
    - Uses asyncio.wait_for() around pubsub listener for clean shutdown
    """

    session_id: str
    task_id: str
    session: ServerSession
    docket: Docket
    _running: bool = field(default=False, init=False)
    _listener_task: asyncio.Task[None] | None = field(default=None, init=False)
    _handler_tasks: set[asyncio.Task[Any]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        # Pre-compute channels to avoid substring issues
        self._elicit_request_channel = self.docket.key(
            f"fastmcp:elicit:{self.session_id}:{self.task_id}:request"
        )
        self._elicit_response_channel = self.docket.key(
            f"fastmcp:elicit:{self.session_id}:{self.task_id}:response"
        )
        self._sample_request_channel = self.docket.key(
            f"fastmcp:sample:{self.session_id}:{self.task_id}:request"
        )
        self._sample_response_channel = self.docket.key(
            f"fastmcp:sample:{self.session_id}:{self.task_id}:response"
        )

    async def start(self) -> None:
        """Start listening for requests on Redis channels."""
        if self._running:
            return

        self._running = True
        self._listener_task = asyncio.create_task(
            self._listen_loop(), name=f"forwarder-{self.session_id}-{self.task_id}"
        )
        logger.info(
            f"Started forwarder for session={self.session_id} task={self.task_id}"
        )

    async def stop(self) -> None:
        """Stop the forwarder and clean up resources."""
        if not self._running:
            return

        self._running = False

        # Cancel listener task
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(self._listener_task, timeout=5.0)

        # Wait for any in-flight handler tasks
        if self._handler_tasks:
            _, pending = await asyncio.wait(
                self._handler_tasks, timeout=10.0, return_when=asyncio.ALL_COMPLETED
            )
            for task in pending:
                task.cancel()

        logger.info(
            f"Stopped forwarder for session={self.session_id} task={self.task_id}"
        )

    async def _listen_loop(self) -> None:
        """Main listening loop for Redis Pub/Sub messages."""
        try:
            async with self.docket.redis() as redis:
                pubsub = redis.pubsub()
                await pubsub.subscribe(
                    self._elicit_request_channel, self._sample_request_channel
                )

                try:
                    async for msg in pubsub.listen():
                        if not self._running:
                            break

                        if msg["type"] != "message":
                            continue

                        channel = msg["channel"]
                        if isinstance(channel, bytes):
                            channel = channel.decode("utf-8")

                        # Exact channel matching - critical for correctness
                        if channel == self._elicit_request_channel:
                            task = asyncio.create_task(
                                self._handle_elicit_request(msg["data"])
                            )
                            self._handler_tasks.add(task)
                            task.add_done_callback(self._handler_tasks.discard)
                        elif channel == self._sample_request_channel:
                            task = asyncio.create_task(
                                self._handle_sample_request(msg["data"])
                            )
                            self._handler_tasks.add(task)
                            task.add_done_callback(self._handler_tasks.discard)

                finally:
                    await pubsub.unsubscribe()
                    await pubsub.close()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Forwarder listener error: {e}")

    async def _handle_elicit_request(self, data: bytes | str) -> None:
        """Handle a single elicitation request."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        try:
            request_data = json.loads(data)
            request_id = request_data["request_id"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid elicit request: {e}")
            return

        try:
            # Import here to avoid circular imports
            import anyio
            import mcp.shared.exceptions
            import mcp.shared.message
            import mcp.types

            from fastmcp.server.tasks.subscriptions import (
                send_input_required_notification,
            )

            # Send input_required status per SEP-1686
            await send_input_required_notification(
                session=self.session,
                task_id=self.task_id,
                session_id=self.session_id,
                docket=self.docket,
                status="input_required",
            )

            try:
                # Use _build_elicit_form_request to emit related-task metadata
                # This matches embedded mode behavior in TaskContext._elicit_embedded
                request = self.session._build_elicit_form_request(  # pyright: ignore[reportPrivateUsage]
                    message=request_data["message"],
                    requestedSchema=request_data.get("schema", {}),
                    related_task_id=self.task_id,
                )

                # Send request and wait for response
                response_stream, response_stream_reader = (
                    anyio.create_memory_object_stream[
                        mcp.types.JSONRPCResponse | mcp.types.JSONRPCError
                    ](1)
                )
                req_id = request.id
                self.session._response_streams[req_id] = response_stream  # pyright: ignore[reportPrivateUsage]

                try:
                    await self.session._write_stream.send(  # pyright: ignore[reportPrivateUsage]
                        mcp.shared.message.SessionMessage(
                            message=mcp.types.JSONRPCMessage(request)
                        )
                    )

                    response_or_error = await response_stream_reader.receive()

                    if isinstance(response_or_error, mcp.types.JSONRPCError):
                        raise mcp.shared.exceptions.McpError(response_or_error.error)
                    result = mcp.types.ElicitResult.model_validate(
                        response_or_error.result
                    )
                finally:
                    self.session._response_streams.pop(req_id, None)  # pyright: ignore[reportPrivateUsage]
                    await response_stream.aclose()
                    await response_stream_reader.aclose()

                response_payload = json.dumps(
                    {
                        "request_id": request_id,
                        "action": result.action,
                        "content": result.content,
                    }
                )
            finally:
                # Restore status to working per SEP-1686
                with suppress(Exception):
                    await send_input_required_notification(
                        session=self.session,
                        task_id=self.task_id,
                        session_id=self.session_id,
                        docket=self.docket,
                        status="working",
                    )

        except Exception as e:
            logger.error(f"Elicit forwarding failed: {e}")
            response_payload = json.dumps(
                {"request_id": request_id, "action": "cancel", "content": None}
            )

        # Publish response
        try:
            async with self.docket.redis() as redis:
                await redis.publish(self._elicit_response_channel, response_payload)
        except Exception as e:
            logger.error(f"Failed to publish elicit response: {e}")

    async def _handle_sample_request(self, data: bytes | str) -> None:
        """Handle a single sampling request."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        try:
            request_data = json.loads(data)
            request_id = request_data["request_id"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid sample request: {e}")
            return

        try:
            # Import here to avoid circular imports
            import anyio
            import mcp.shared.exceptions
            import mcp.shared.message
            import mcp.types
            from pydantic import TypeAdapter

            from fastmcp.server.tasks.subscriptions import (
                send_input_required_notification,
            )

            # Build sampling messages using TypeAdapter for proper content type handling
            # This supports text, image, audio, and any future content types
            content_adapter = TypeAdapter(mcp.types.SamplingContent)
            messages: list[mcp.types.SamplingMessage] = []
            for m in request_data["messages"]:
                role = m.get("role", "user")
                raw_content = m.get("content", {})
                if isinstance(raw_content, str):
                    content = mcp.types.TextContent(type="text", text=raw_content)
                else:
                    content = content_adapter.validate_python(raw_content)
                messages.append(mcp.types.SamplingMessage(role=role, content=content))

            # Send input_required status per SEP-1686
            await send_input_required_notification(
                session=self.session,
                task_id=self.task_id,
                session_id=self.session_id,
                docket=self.docket,
                status="input_required",
            )

            try:
                # Use _build_create_message_request to emit related-task metadata
                # This matches embedded mode behavior in TaskContext._sample_embedded
                request = self.session._build_create_message_request(  # pyright: ignore[reportPrivateUsage]
                    messages=messages,
                    max_tokens=request_data.get("max_tokens", 512),
                    system_prompt=request_data.get("system_prompt"),
                    temperature=request_data.get("temperature"),
                    model_preferences=request_data.get("model_preferences"),
                    related_task_id=self.task_id,
                )

                # Send request and wait for response
                response_stream, response_stream_reader = (
                    anyio.create_memory_object_stream[
                        mcp.types.JSONRPCResponse | mcp.types.JSONRPCError
                    ](1)
                )
                req_id = request.id
                self.session._response_streams[req_id] = response_stream  # pyright: ignore[reportPrivateUsage]

                try:
                    await self.session._write_stream.send(  # pyright: ignore[reportPrivateUsage]
                        mcp.shared.message.SessionMessage(
                            message=mcp.types.JSONRPCMessage(request)
                        )
                    )

                    response_or_error = await response_stream_reader.receive()

                    if isinstance(response_or_error, mcp.types.JSONRPCError):
                        raise mcp.shared.exceptions.McpError(response_or_error.error)
                    result = mcp.types.CreateMessageResult.model_validate(
                        response_or_error.result
                    )
                finally:
                    self.session._response_streams.pop(req_id, None)  # pyright: ignore[reportPrivateUsage]
                    await response_stream.aclose()
                    await response_stream_reader.aclose()

            finally:
                # Restore status to working per SEP-1686
                with suppress(Exception):
                    await send_input_required_notification(
                        session=self.session,
                        task_id=self.task_id,
                        session_id=self.session_id,
                        docket=self.docket,
                        status="working",
                    )

            # Serialize result
            response_payload = json.dumps(
                {
                    "request_id": request_id,
                    "result": result.model_dump(mode="json"),
                }
            )

        except Exception as e:
            logger.error(f"Sample forwarding failed: {e}")
            response_payload = json.dumps({"request_id": request_id, "error": str(e)})

        # Publish response
        try:
            async with self.docket.redis() as redis:
                await redis.publish(self._sample_response_channel, response_payload)
        except Exception as e:
            logger.error(f"Failed to publish sample response: {e}")


# Registry of active forwarders - uses WeakValueDictionary for automatic cleanup
# when session references are dropped
_active_forwarders: dict[ForwarderKey, ElicitForwarder] = {}


async def start_forwarder(
    session_id: str, task_id: str, session: ServerSession, docket: Docket
) -> ElicitForwarder | None:
    """Start a forwarder for the given session/task pair.

    Returns None if distributed mode is not enabled.

    Args:
        session_id: The session ID
        task_id: The MCP task ID
        session: The ServerSession to forward to
        docket: Docket instance with Redis connection

    Returns:
        ElicitForwarder instance if started, None if disabled
    """
    if not is_distributed_mode_enabled():
        return None

    key = ForwarderKey(session_id, task_id)

    if key in _active_forwarders:
        logger.debug(f"Forwarder already exists for {key}")
        return _active_forwarders[key]

    forwarder = ElicitForwarder(
        session_id=session_id, task_id=task_id, session=session, docket=docket
    )
    _active_forwarders[key] = forwarder

    await forwarder.start()
    return forwarder


async def stop_forwarder(session_id: str, task_id: str) -> None:
    """Stop a specific forwarder."""
    key = ForwarderKey(session_id, task_id)
    forwarder = _active_forwarders.pop(key, None)
    if forwarder:
        await forwarder.stop()


async def stop_forwarders_for_session(session_id: str) -> None:
    """Stop all forwarders associated with a session.

    Called when a session disconnects.
    """
    keys_to_remove = [k for k in _active_forwarders if k.session_id == session_id]
    for key in keys_to_remove:
        forwarder = _active_forwarders.pop(key)
        await forwarder.stop()


async def stop_all_forwarders() -> None:
    """Stop all active forwarders.

    Used during server shutdown.
    """
    for key in list(_active_forwarders.keys()):
        forwarder = _active_forwarders.pop(key)
        await forwarder.stop()

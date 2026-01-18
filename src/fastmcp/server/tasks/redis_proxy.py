"""Redis-based proxy for TaskContext operations in distributed workers.

This module provides functions for distributed workers to communicate with the
FastMCP server process via Redis Pub/Sub. When a worker needs to elicit user
input or request sampling, it publishes a request to Redis and waits for the
server's response.

Note:
    Redis Pub/Sub is a "fire and forget" mechanism—messages are not persisted.
    This design relies on request-level timeouts to detect message loss.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastmcp.utilities.logging import get_logger

if TYPE_CHECKING:
    from docket import Docket

logger = get_logger(__name__)

# Environment-configurable timeouts
ELICIT_TIMEOUT_SECONDS = int(os.getenv("FASTMCP_ELICIT_TIMEOUT", "300"))  # 5 min
SAMPLE_TIMEOUT_SECONDS = int(os.getenv("FASTMCP_SAMPLE_TIMEOUT", "120"))  # 2 min


@dataclass
class ElicitRequest:
    """Elicitation request sent from worker to server."""

    request_id: str
    message: str
    schema: dict[str, Any]
    task_id: str
    session_id: str


@dataclass
class ElicitResponse:
    """Elicitation response sent from server to worker."""

    request_id: str
    action: str  # "accept", "decline", "cancel"
    content: dict[str, Any] | None


async def send_elicit_via_redis(
    docket: Docket,
    session_id: str,
    task_id: str,
    message: str,
    schema: dict[str, Any],
    timeout: float = ELICIT_TIMEOUT_SECONDS,
) -> ElicitResponse:
    """Send elicitation request via Redis and wait for response.

    This is used by distributed workers that cannot directly access
    the ServerSession.

    Args:
        docket: Docket instance with Redis connection
        session_id: The session ID for this task
        task_id: The MCP task ID
        message: The elicitation message
        schema: The JSON schema for expected response
        timeout: Timeout in seconds

    Returns:
        ElicitResponse with action and content

    Raises:
        TimeoutError: If no response within timeout
        RuntimeError: If Redis connection fails
    """
    request_id = str(uuid.uuid4())
    request_channel = docket.key(f"fastmcp:elicit:{session_id}:{task_id}:request")
    response_channel = docket.key(f"fastmcp:elicit:{session_id}:{task_id}:response")

    request_payload = json.dumps(
        {
            "request_id": request_id,
            "message": message,
            "schema": schema,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    try:
        async with docket.redis() as redis:
            # Subscribe to response channel BEFORE publishing request
            # to avoid race condition
            pubsub = redis.pubsub()
            await pubsub.subscribe(response_channel)

            try:
                # Publish the request
                await redis.publish(request_channel, request_payload)
                logger.debug(
                    f"Published elicit request {request_id} to {request_channel}"
                )

                # Define the listener coroutine
                async def _listen_for_response() -> ElicitResponse:
                    async for msg in pubsub.listen():
                        if msg["type"] != "message":
                            continue

                        try:
                            data = msg["data"]
                            if isinstance(data, bytes):
                                data = data.decode("utf-8")
                            response_data = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        # Check if this response is for our request
                        if response_data.get("request_id") != request_id:
                            continue

                        return ElicitResponse(
                            request_id=request_id,
                            action=response_data["action"],
                            content=response_data.get("content"),
                        )

                    raise RuntimeError("Pub/sub listener ended unexpectedly")

                # Wrap entire listener with timeout - this ensures timeout fires
                # even if pubsub.listen() is blocking waiting for messages
                try:
                    return await asyncio.wait_for(
                        _listen_for_response(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"Elicitation request {request_id} timed out after {timeout}s"
                    ) from None

            finally:
                await pubsub.unsubscribe(response_channel)
                await pubsub.close()

    except Exception as e:
        if isinstance(e, TimeoutError):
            raise
        raise RuntimeError(
            f"Redis connection failed during distributed elicitation. "
            f"Ensure FASTMCP_DOCKET_URL points to a running Redis instance. "
            f"Original error: {e}"
        ) from e


async def send_sample_via_redis(
    docket: Docket,
    session_id: str,
    task_id: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 512,
    system_prompt: str | None = None,
    temperature: float | None = None,
    timeout: float = SAMPLE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Send sampling request via Redis and wait for response.

    Args:
        docket: Docket instance with Redis connection
        session_id: The session ID for this task
        task_id: The MCP task ID
        messages: Sampling messages
        max_tokens: Maximum tokens in response
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        timeout: Timeout in seconds

    Returns:
        CreateMessageResult as dict

    Raises:
        TimeoutError: If no response within timeout
        RuntimeError: If Redis connection or sampling fails
    """
    request_id = str(uuid.uuid4())
    request_channel = docket.key(f"fastmcp:sample:{session_id}:{task_id}:request")
    response_channel = docket.key(f"fastmcp:sample:{session_id}:{task_id}:response")

    request_payload = json.dumps(
        {
            "request_id": request_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    try:
        async with docket.redis() as redis:
            pubsub = redis.pubsub()
            await pubsub.subscribe(response_channel)

            try:
                await redis.publish(request_channel, request_payload)
                logger.debug(
                    f"Published sample request {request_id} to {request_channel}"
                )

                # Define the listener coroutine
                async def _listen_for_response() -> dict[str, Any]:
                    async for msg in pubsub.listen():
                        if msg["type"] != "message":
                            continue

                        try:
                            data = msg["data"]
                            if isinstance(data, bytes):
                                data = data.decode("utf-8")
                            response_data = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if response_data.get("request_id") != request_id:
                            continue

                        # Check for error
                        if "error" in response_data:
                            raise RuntimeError(response_data["error"])

                        return response_data["result"]

                    raise RuntimeError("Pub/sub listener ended unexpectedly")

                # Wrap entire listener with timeout
                try:
                    return await asyncio.wait_for(
                        _listen_for_response(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"Sampling request {request_id} timed out after {timeout}s"
                    ) from None

            finally:
                await pubsub.unsubscribe(response_channel)
                await pubsub.close()

    except Exception as e:
        if isinstance(e, (TimeoutError, RuntimeError)):
            raise
        raise RuntimeError(
            f"Redis connection failed during distributed sampling. "
            f"Ensure FASTMCP_DOCKET_URL points to a running Redis instance. "
            f"Original error: {e}"
        ) from e

"""Tests for distributed TaskContext implementation.

These tests verify the Redis-based proxy and forwarder for distributed workers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fastmcp.server.tasks.forwarder import (
    ForwarderKey,
    start_forwarder,
    stop_all_forwarders,
    stop_forwarder,
    stop_forwarders_for_session,
)
from fastmcp.server.tasks.redis_proxy import (
    ElicitRequest,
    ElicitResponse,
)


class TestDistributedModeDetection:
    """Tests for FASTMCP_DISTRIBUTED_WORKERS feature flag."""

    def test_distributed_mode_disabled_by_default(self) -> None:
        """Distributed mode should be disabled when env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Clear cache
            import importlib

            import fastmcp.server.tasks.forwarder as forwarder_module

            importlib.reload(forwarder_module)
            assert not forwarder_module.is_distributed_mode_enabled()

    def test_distributed_mode_enabled_with_1(self) -> None:
        """Distributed mode enabled with FASTMCP_DISTRIBUTED_WORKERS=1."""
        with patch.dict("os.environ", {"FASTMCP_DISTRIBUTED_WORKERS": "1"}):
            import importlib

            import fastmcp.server.tasks.forwarder as forwarder_module

            importlib.reload(forwarder_module)
            assert forwarder_module.is_distributed_mode_enabled()

    def test_distributed_mode_enabled_with_true(self) -> None:
        """Distributed mode enabled with FASTMCP_DISTRIBUTED_WORKERS=true."""
        with patch.dict("os.environ", {"FASTMCP_DISTRIBUTED_WORKERS": "true"}):
            import importlib

            import fastmcp.server.tasks.forwarder as forwarder_module

            importlib.reload(forwarder_module)
            assert forwarder_module.is_distributed_mode_enabled()


class TestForwarderKey:
    """Tests for ForwarderKey dataclass."""

    def test_forwarder_key_equality(self) -> None:
        """ForwarderKey should be hashable and comparable."""
        key1 = ForwarderKey(session_id="sess1", task_id="task1")
        key2 = ForwarderKey(session_id="sess1", task_id="task1")
        key3 = ForwarderKey(session_id="sess2", task_id="task1")

        assert key1 == key2
        assert key1 != key3
        assert hash(key1) == hash(key2)


class TestElicitRequest:
    """Tests for ElicitRequest dataclass."""

    def test_elicit_request_fields(self) -> None:
        """ElicitRequest should have all required fields."""
        request = ElicitRequest(
            request_id="req1",
            message="What is your name?",
            schema={"type": "string"},
            task_id="task1",
            session_id="sess1",
        )
        assert request.request_id == "req1"
        assert request.message == "What is your name?"
        assert request.schema == {"type": "string"}


class TestElicitResponse:
    """Tests for ElicitResponse dataclass."""

    def test_elicit_response_accept(self) -> None:
        """ElicitResponse should handle accept action."""
        response = ElicitResponse(
            request_id="req1",
            action="accept",
            content={"name": "Alice"},
        )
        assert response.action == "accept"
        assert response.content == {"name": "Alice"}

    def test_elicit_response_decline(self) -> None:
        """ElicitResponse should handle decline action."""
        response = ElicitResponse(
            request_id="req1",
            action="decline",
            content=None,
        )
        assert response.action == "decline"
        assert response.content is None


class TestForwarderLifecycle:
    """Tests for forwarder start/stop lifecycle."""

    @pytest.mark.anyio
    async def test_start_forwarder_returns_none_when_disabled(self) -> None:
        """start_forwarder should return None when distributed mode is disabled."""
        with patch(
            "fastmcp.server.tasks.forwarder.is_distributed_mode_enabled",
            return_value=False,
        ):
            session = MagicMock()
            docket = MagicMock()

            result = await start_forwarder(
                session_id="sess1",
                task_id="task1",
                session=session,
                docket=docket,
            )
            assert result is None

    @pytest.mark.anyio
    async def test_stop_forwarder_handles_missing(self) -> None:
        """stop_forwarder should handle non-existent forwarder gracefully."""
        # Should not raise
        await stop_forwarder("nonexistent", "nonexistent")

    @pytest.mark.anyio
    async def test_stop_forwarders_for_session_handles_empty(self) -> None:
        """stop_forwarders_for_session should handle no forwarders gracefully."""
        # Should not raise
        await stop_forwarders_for_session("nonexistent")

    @pytest.mark.anyio
    async def test_stop_all_forwarders_handles_empty(self) -> None:
        """stop_all_forwarders should handle no forwarders gracefully."""
        # Should not raise
        await stop_all_forwarders()


class TestTaskContextDistributedMode:
    """Tests for TaskContext distributed mode detection."""

    def test_task_context_embedded_mode(self) -> None:
        """TaskContext should detect embedded mode when session is available."""
        from fastmcp.server.dependencies import TaskContext, register_task_session

        # Create a mock session
        mock_session = MagicMock()
        register_task_session("sess1", mock_session)

        try:
            ctx = TaskContext(task_id="task1", session_id="sess1")
            assert not ctx.is_distributed
            assert ctx.session_available
        finally:
            # Cleanup
            from fastmcp.server.dependencies import _task_sessions

            _task_sessions.pop("sess1", None)

    def test_task_context_distributed_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TaskContext should detect distributed mode when session is not available."""
        from fastmcp.server.dependencies import TaskContext

        # Enable distributed mode via feature flag
        monkeypatch.setenv("FASTMCP_DISTRIBUTED_WORKERS", "1")

        # No session registered, feature flag enabled
        ctx = TaskContext(task_id="task1", session_id="nonexistent")
        assert ctx.is_distributed
        assert not ctx.session_available

    def test_task_context_is_distributed_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_distributed property should reflect mode and feature flag."""
        from fastmcp.server.dependencies import TaskContext

        # Enable distributed mode via feature flag
        monkeypatch.setenv("FASTMCP_DISTRIBUTED_WORKERS", "1")

        ctx = TaskContext(task_id="task1", session_id="nonexistent")
        assert ctx.is_distributed is True
        assert ctx.session_available is False

        # Verify it's a property, not a method
        assert isinstance(type(ctx).is_distributed, property)
        assert isinstance(type(ctx).session_available, property)

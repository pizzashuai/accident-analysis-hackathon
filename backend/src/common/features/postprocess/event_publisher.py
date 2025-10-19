#!/usr/bin/env python3
"""
Redis-based event publisher for LLM analysis streaming.

This module provides a publisher that sends events to Redis channels
for real-time streaming of LLM analysis progress to the frontend.
"""

import json
import logging
from datetime import datetime
from typing import Any

import redis

from src.common.config import settings

logger = logging.getLogger(__name__)


class LLMEventPublisher:
    """
    Publishes LLM analysis events to Redis for real-time streaming.

    Events are published to Redis channels keyed by analysis_id,
    allowing multiple frontend clients to subscribe to the same analysis.
    """

    def __init__(self, redis_url: str | None = None):
        """
        Initialize the event publisher.

        Args:
            redis_url: Redis connection URL. If None, uses default from settings.
        """
        self.redis_url = redis_url or getattr(
            settings, "REDIS_URL", "redis://localhost:6379/0"
        )
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)

        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _publish_event(
        self, analysis_id: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """
        Publish an event to the Redis channel for the given analysis_id.

        Args:
            analysis_id: Unique identifier for the analysis session
            event_type: Type of event (thinking_start, tool_call_start, etc.)
            data: Event data payload
        """
        channel = f"llm_analysis:{analysis_id}"

        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        try:
            self.redis_client.publish(channel, json.dumps(event))
            logger.debug(f"Published {event_type} event to {channel}")
        except Exception as e:
            logger.error(f"Failed to publish event to Redis: {e}")
            raise

    def publish_thinking_start(
        self, analysis_id: str, message: str = "Starting analysis..."
    ) -> None:
        """Publish thinking start event."""
        self._publish_event(analysis_id, "thinking_start", {"message": message})

    def publish_thinking_content(self, analysis_id: str, content: str) -> None:
        """Publish thinking content event (streaming reasoning)."""
        self._publish_event(analysis_id, "thinking_content", {"content": content})

    def publish_thinking_end(
        self, analysis_id: str, message: str = "Analysis complete"
    ) -> None:
        """Publish thinking end event."""
        self._publish_event(analysis_id, "thinking_end", {"message": message})

    def publish_tool_call_start(
        self, analysis_id: str, tool_name: str, tool_input: dict[str, Any]
    ) -> None:
        """Publish tool call start event."""
        self._publish_event(
            analysis_id, "tool_call_start", {"tool": tool_name, "input": tool_input}
        )

    def publish_tool_call_result(
        self, analysis_id: str, tool_name: str, result: dict[str, Any]
    ) -> None:
        """Publish tool call result event."""
        self._publish_event(
            analysis_id, "tool_call_result", {"tool": tool_name, "result": result}
        )

    def publish_report_start(
        self, analysis_id: str, message: str = "Generating final report..."
    ) -> None:
        """Publish report generation start event."""
        self._publish_event(analysis_id, "report_start", {"message": message})

    def publish_report_content(self, analysis_id: str, content: str) -> None:
        """Publish report content event (streaming report)."""
        self._publish_event(analysis_id, "report_content", {"content": content})

    def publish_report_end(
        self, analysis_id: str, message: str = "Report complete"
    ) -> None:
        """Publish report end event."""
        self._publish_event(analysis_id, "report_end", {"message": message})

    def publish_error(
        self,
        analysis_id: str,
        error_message: str,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        """Publish error event."""
        data = {"error": error_message}
        if error_details:
            data["details"] = error_details
        self._publish_event(analysis_id, "error", data)

    def publish_model_switch(
        self, analysis_id: str, old_model: str, new_model: str
    ) -> None:
        """Publish model switch event (for fallback scenarios)."""
        self._publish_event(
            analysis_id,
            "model_switch",
            {
                "old_model": old_model,
                "new_model": new_model,
                "message": f"Switching from {old_model} to {new_model}",
            },
        )

    def publish_iteration_update(
        self, analysis_id: str, iteration: int, max_iterations: int
    ) -> None:
        """Publish iteration update event."""
        self._publish_event(
            analysis_id,
            "iteration_update",
            {
                "iteration": iteration,
                "max_iterations": max_iterations,
                "message": f"Iteration {iteration}/{max_iterations}",
            },
        )

    def publish_collision_detected(
        self, analysis_id: str, message: str = "Collision detected!"
    ) -> None:
        """Publish collision detection event."""
        self._publish_event(analysis_id, "collision_detected", {"message": message})

    def close(self) -> None:
        """Close the Redis connection."""
        if hasattr(self, "redis_client"):
            self.redis_client.close()

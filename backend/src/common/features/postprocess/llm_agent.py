#!/usr/bin/env python3
"""
LLM-Powered Accident Analysis Agent.
This agent uses AWS Bedrock Claude to analyze accident data and generate flexible reports.
The LLM decides the output format based on the data characteristics.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import boto3
from botocore.exceptions import ClientError

from ...config import settings
from .agent_tools import AgentToolHandler
from .event_publisher import LLMEventPublisher

logger = logging.getLogger(__name__)
@dataclass
class LLMAgentConfig:
    """Configuration for the LLM-powered accident analysis agent."""

    track_ids: list[int]
    frame_range: tuple[int, int] | None = None
    iou_threshold: float = 0.01
    distance_threshold_m: float = 5.0
    persistence_frames: int = 3
    padding_frames: int = 10
    detections_file: str = "../process_video/detections.jsonl"
    aws_region: str = "us-west-2"
    bedrock_models: list[str] = None  # type: ignore
    max_iterations: int = 20
    temperature: float = 0.0
    max_tokens: int = 4096

    def __post_init__(self):
        """Set default models if none provided."""
        if self.bedrock_models is None:
            self.bedrock_models = [
                "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            ]
        # Ensure bedrock_models is always a list for type safety
        assert isinstance(self.bedrock_models, list), "bedrock_models must be a list"


class LLMAccidentAnalysisAgent:
    """
    LLM-powered accident analysis agent that uses AWS Bedrock Claude.

    This agent:
    1. Loads detection data for specified track IDs
    2. Uses Claude to analyze the data and determine appropriate tools
    3. Generates flexible reports based on data characteristics
    4. Lets the LLM decide the output format and structure
    """

    SYSTEM_PROMPT = """You are an expert accident reconstruction analyst specializing in vehicle collision analysis from video detection data.

Your task is to analyze vehicle detection data and generate comprehensive accident analysis reports using everyday language that anyone can understand. You have access to specialized tools for data analysis.

CRITICAL REQUIREMENTS:
- Write in clear, everyday language that non-technical people can understand
- Focus on describing what happened rather than technical calculations
- Include specific timestamps and frame numbers for all events
- Create a chronological timeline of events with clear descriptions
- Minimize technical jargon and visual calculations
- Use descriptive language about vehicle behavior and interactions
- ALL DISTANCE MEASUREMENTS MUST BE IN MILES
- ALL SPEED MEASUREMENTS MUST BE IN MILES PER HOUR (MPH)

AVAILABLE TOOLS:
1. load_detections - Load detection data for specified track IDs
2. compute_pair_metrics - Calculate collision metrics (IoU, distances, speeds)
3. trace_impact_window - Detect collision events and impact windows
4. build_timeline - Generate structured event timeline
5. report_assumptions - Identify data quality issues

ANALYSIS WORKFLOW:
1. Start by loading the detection data for the specified track IDs
2. Compute collision metrics to understand vehicle interactions
3. Trace impact windows to determine if collision occurred
4. Build timeline of events with specific timestamps and descriptions
5. Report data quality issues and assumptions
6. Generate a comprehensive analysis report in everyday language

REPORT GENERATION:
- Write in clear, conversational language that tells the story of what happened
- Focus on describing vehicle movements and speeds
- Include timestamps for all significant events
- Describe the sequence of events chronologically
- Minimize technical calculations and focus on narrative descriptions
- Include specific frame numbers and timestamps for reference
- Highlight any data limitations or assumptions in simple terms
- Provide clear conclusions about what occurred

Generate a professional, easy-to-understand accident analysis report that tells the story of what happened."""

    def __init__(
        self,
        config: LLMAgentConfig,
        event_publisher: LLMEventPublisher | None = None,
        analysis_id: str | None = None,
    ):
        """Initialize the LLM agent."""
        self.config = config
        self.tool_handler = AgentToolHandler(detections_file=config.detections_file)
        self.conversation_history: list[dict[str, Any]] = []
        self.execution_log: list[dict[str, Any]] = []

        # Event publishing
        self.event_publisher = event_publisher
        self.analysis_id = analysis_id

        # Model management
        self.current_model_index = 0
        self.current_model_id = config.bedrock_models[0]
        self.model_attempts = {}  # Track attempts per model

        # Initialize AWS Bedrock client
        self.bedrock_client = self._initialize_bedrock_client()

    def _publish_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event if event publisher is available."""
        if self.event_publisher and self.analysis_id:
            try:
                logger.info(f"Publishing event {event_type} for analysis {self.analysis_id}: {data}")
                if event_type == "thinking_start":
                    self.event_publisher.publish_thinking_start(
                        self.analysis_id, data.get("message", "Starting analysis...")
                    )
                elif event_type == "thinking_content":
                    self.event_publisher.publish_thinking_content(
                        self.analysis_id, data["content"]
                    )
                elif event_type == "thinking_end":
                    self.event_publisher.publish_thinking_end(
                        self.analysis_id, data.get("message", "Analysis complete")
                    )
                elif event_type == "tool_call_start":
                    self.event_publisher.publish_tool_call_start(
                        self.analysis_id, data["tool"], data["input"]
                    )
                elif event_type == "tool_call_result":
                    self.event_publisher.publish_tool_call_result(
                        self.analysis_id, data["tool"], data["result"]
                    )
                elif event_type == "report_start":
                    self.event_publisher.publish_report_start(
                        self.analysis_id,
                        data.get("message", "Generating final report..."),
                    )
                elif event_type == "report_content":
                    self.event_publisher.publish_report_content(
                        self.analysis_id, data["content"]
                    )
                elif event_type == "report_end":
                    self.event_publisher.publish_report_end(
                        self.analysis_id, data.get("message", "Report complete")
                    )
                elif event_type == "error":
                    self.event_publisher.publish_error(
                        self.analysis_id, data["error"], data.get("details")
                    )
                elif event_type == "model_switch":
                    self.event_publisher.publish_model_switch(
                        self.analysis_id, data["old_model"], data["new_model"]
                    )
                elif event_type == "iteration_update":
                    self.event_publisher.publish_iteration_update(
                        self.analysis_id, data["iteration"], data["max_iterations"]
                    )
                elif event_type == "collision_detected":
                    self.event_publisher.publish_collision_detected(
                        self.analysis_id, data.get("message", "Collision detected!")
                    )
                logger.info(f"Successfully published event {event_type}")
            except Exception as e:
                logger.error(f"Failed to publish event {event_type}: {e}")
                print(f"Warning: Failed to publish event {event_type}: {e}")
        else:
            logger.warning(f"No event publisher or analysis_id available for event {event_type}")

    def _is_throttling_error(self, error: Exception) -> bool:
        """Check if the error is a throttling error."""
        error_str = str(error).lower()
        throttling_indicators = [
            "throttle",
            "rate limit",
            "too many requests",
            "quota exceeded",
            "service unavailable",
            "throttlingexception",
            "request limit exceeded",
        ]
        return any(indicator in error_str for indicator in throttling_indicators)

    def _switch_to_next_model(self) -> bool:
        """Switch to the next available model. Returns True if successful, False if no more models."""
        old_model = self.current_model_id
        self.current_model_index += 1

        if self.current_model_index >= len(self.config.bedrock_models):
            print(
                f"\n❌ All models exhausted. Tried {len(self.config.bedrock_models)} models."
            )
            return False

        self.current_model_id = self.config.bedrock_models[self.current_model_index]

        # Initialize attempt counter for the new model
        if self.current_model_id not in self.model_attempts:
            self.model_attempts[self.current_model_id] = 0

        print(f"\n🔄 Switching to model: {self.current_model_id}")

        # Publish model switch event
        self._publish_event(
            "model_switch", {"old_model": old_model, "new_model": self.current_model_id}
        )

        return True

    def _initialize_bedrock_client(self):
        """Initialize AWS Bedrock client with credentials."""
        try:
            # Try to get credentials from environment or AWS config
            client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.config.aws_region,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )

            # Test the connection by trying to list foundation models
            try:
                client.list_foundation_models()
            except AttributeError:
                # Some versions of boto3 don't have this method, that's OK
                pass
            return client

        except ClientError as e:
            print(f"Warning: Could not initialize Bedrock client: {e}")
            print(
                "Make sure AWS credentials are configured (aws configure or environment variables)"
            )
            return None
        except Exception as e:
            print(f"Warning: Unexpected error initializing Bedrock: {e}")
            return None

    def analyze(self) -> dict[str, Any]:
        """
        Execute LLM-powered accident analysis.

        Returns:
            Complete analysis results with LLM-generated report
        """
        if not self.bedrock_client:
            error_msg = (
                "Bedrock client not available. Please configure AWS credentials."
            )
            self._publish_event("error", {"error": error_msg})
            return {
                "success": False,
                "error": error_msg,
                "suggestion": "Run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables",
            }

        # Publish analysis start event
        self._publish_event(
            "thinking_start", {"message": "Starting accident analysis..."}
        )

        print("=" * 80)
        print("LLM-POWERED ACCIDENT ANALYSIS")
        print("=" * 80)
        print(f"Analyzing tracks: {self.config.track_ids}")
        print(f"Available models: {', '.join(self.config.bedrock_models)}")
        print(f"Starting with model: {self.current_model_id}")
        print(f"AWS Region: {self.config.aws_region}")
        print("-" * 80)

        # Create initial query
        initial_query = self._create_initial_query()
        self.conversation_history.append(
            {"role": "user", "content": [{"text": initial_query}]}
        )

        print(f"\n[INITIAL QUERY]\n{initial_query}\n")

        # Run agentic loop
        iteration = 0
        final_report = None

        while iteration < self.config.max_iterations:
            iteration += 1
            print(f"\n[ITERATION {iteration}]")

            # Publish iteration update
            self._publish_event(
                "iteration_update",
                {"iteration": iteration, "max_iterations": self.config.max_iterations},
            )

            # Call Claude with tool use capability
            response = self._call_bedrock_with_tools()

            # Check if we got a final answer
            if self._is_final_answer(response):
                self._publish_event(
                    "report_start", {"message": "Generating final report..."}
                )
                final_report = self._extract_final_report(response)
                print("\n✓ Analysis complete - LLM generated final report")
                break

            # Process tool use requests
            tool_results = self._process_tool_use(response)

            if not tool_results:
                print("\n⚠ No tool use detected, ending iteration")
                break

            # Add tool results to conversation
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": response.get("output", {})
                    .get("message", {})
                    .get("content", []),
                }
            )

            self.conversation_history.append({"role": "user", "content": tool_results})

        if not final_report:
            print("\n⚠ Max iterations reached without final report")
            self._publish_event(
                "report_start",
                {"message": "Requesting final report after max iterations..."},
            )
            final_report = self._request_final_report()

        # Publish thinking end and report content
        self._publish_event("thinking_end", {"message": "Analysis reasoning complete"})

        # Stream the report content in chunks
        if final_report:
            self._stream_report_content(final_report)

        # Format and return the report
        return self._format_llm_report(final_report, iteration)

    def _stream_report_content(self, report_text: str) -> None:
        """Stream report content in chunks for real-time display."""
        if not report_text:
            return

        # Split report into chunks for streaming effect
        chunk_size = 100  # Characters per chunk
        chunks = [
            report_text[i : i + chunk_size]
            for i in range(0, len(report_text), chunk_size)
        ]

        for chunk in chunks:
            self._publish_event("report_content", {"content": chunk})
            # Small delay to simulate streaming (optional)
            import time

            time.sleep(0.05)  # 50ms delay between chunks

        # Publish report end
        self._publish_event("report_end", {"message": "Report generation complete"})

    def _create_initial_query(self) -> str:
        """Create the initial query for the LLM."""
        query = f"""Please analyze the vehicle collision data for tracks {self.config.track_ids}.

Configuration:
- Track IDs: {self.config.track_ids}
- Frame range: {self.config.frame_range if self.config.frame_range else "All frames"}
- IoU threshold: {self.config.iou_threshold}
- Distance threshold: {self.config.distance_threshold_m}m
- Persistence frames: {self.config.persistence_frames}
- Detections file: {self.config.detections_file}

Please:
1. Use the available tools to load and analyze the detection data
2. Determine if a collision occurred or if this was a near-miss
3. Generate a comprehensive analysis report with appropriate format based on the data
4. Include specific frame citations, timestamps, and metrics
5. Highlight any data quality issues or assumptions
6. IMPORTANT: All distance measurements must be in miles and all speed measurements must be in miles per hour (MPH)

Start by loading the detection data for the specified tracks."""
        return query

    def _call_bedrock_with_tools(self) -> dict[str, Any]:
        """Call AWS Bedrock Claude with tool use capability and automatic model fallback."""
        if not self.bedrock_client:
            raise RuntimeError("Bedrock client not initialized")

        # Prepare tool definitions for Claude
        tools = self._prepare_tool_definitions()

        # Prepare messages
        messages = self.conversation_history.copy()

        # Track attempts for current model
        if self.current_model_id not in self.model_attempts:
            self.model_attempts[self.current_model_id] = 0
        self.model_attempts[self.current_model_id] += 1

        # Call Bedrock with retry logic
        while True:
            try:
                print(
                    f"  → Calling model: {self.current_model_id} (attempt {self.model_attempts[self.current_model_id]})"
                )

                response = self.bedrock_client.converse(
                    modelId=self.current_model_id,
                    messages=messages,
                    system=[{"text": self.SYSTEM_PROMPT}],
                    toolConfig={"tools": tools},
                    inferenceConfig={
                        "maxTokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                    },
                )

                return response

            except Exception as e:
                print(
                    f"\n✗ Error calling Bedrock with {self.current_model_id}: {str(e)}"
                )

                # Check if it's a throttling error
                if self._is_throttling_error(e):
                    print(f"  → Throttling detected for {self.current_model_id}")

                    # Try to switch to next model
                    if self._switch_to_next_model():
                        continue  # Retry with new model
                    else:
                        # All models exhausted
                        raise RuntimeError(
                            f"All models exhausted due to throttling. Last error: {str(e)}"
                        )
                else:
                    # Non-throttling error, re-raise immediately
                    raise

    def _prepare_tool_definitions(self) -> list[dict[str, Any]]:
        """Prepare tool definitions in Bedrock format."""
        tools = []

        # Define load_detections tool
        tools.append(
            {
                "toolSpec": {
                    "name": "load_detections",
                    "description": "Load detection data for specified track IDs from the detections JSONL file. Returns paired detection records sorted by frame with metadata about data quality.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "track_ids": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "List of track IDs to load (e.g., [7, 14])",
                                },
                                "frame_range": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "Optional [start_frame, end_frame] to restrict loading",
                                },
                                "require_pairing": {
                                    "type": "boolean",
                                    "description": "If true, only return frames where all track_ids appear",
                                    "default": True,
                                },
                            },
                            "required": ["track_ids"],
                        }
                    },
                }
            }
        )

        # Define compute_pair_metrics tool
        tools.append(
            {
                "toolSpec": {
                    "name": "compute_pair_metrics",
                    "description": "Compute collision metrics (IoU, distances, speeds) for paired detections. Returns enriched data with flags for collision candidates.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "iou_threshold": {
                                    "type": "number",
                                    "description": "IoU threshold for collision detection",
                                    "default": 0.01,
                                },
                                "vehicle_length_m": {
                                    "type": "number",
                                    "description": "Assumed vehicle length for distance calculations",
                                    "default": 4.5,
                                },
                                "include_headings": {
                                    "type": "boolean",
                                    "description": "Whether to compute heading differences",
                                    "default": False,
                                },
                            },
                            "required": [],
                        }
                    },
                }
            }
        )

        # Define trace_impact_window tool
        tools.append(
            {
                "toolSpec": {
                    "name": "trace_impact_window",
                    "description": "Trace the impact window to detect collision events. Returns collision detection results, impact frames, and diagnostic notes.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "iou_threshold": {
                                    "type": "number",
                                    "description": "IoU threshold for collision detection",
                                    "default": 0.01,
                                },
                                "distance_threshold_m": {
                                    "type": "number",
                                    "description": "Distance threshold in meters",
                                    "default": 5.0,
                                },
                                "persistence_frames": {
                                    "type": "number",
                                    "description": "Number of frames to persist overlap",
                                    "default": 3,
                                },
                            },
                            "required": [],
                        }
                    },
                }
            }
        )

        # Define build_timeline tool
        tools.append(
            {
                "toolSpec": {
                    "name": "build_timeline",
                    "description": "Build a structured timeline of events from approach to separation with everyday language descriptions. Returns timeline entries with frame, timestamp, metrics, and narrative.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "padding_frames": {
                                    "type": "number",
                                    "description": "Number of frames to include before/after impact",
                                    "default": 10,
                                },
                                "stages": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Custom stage labels",
                                },
                            },
                            "required": [],
                        }
                    },
                }
            }
        )

        # Define report_assumptions tool
        tools.append(
            {
                "toolSpec": {
                    "name": "report_assumptions",
                    "description": "Report data quality issues and assumptions. Returns list of warnings about missing data, low confidence detections, and frame gaps.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "warn_if_missing": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Fields to warn about if missing",
                                    "default": ["world_coords", "speed_mph"],
                                }
                            },
                            "required": [],
                        }
                    },
                }
            }
        )

        return tools

    def _is_final_answer(self, response: dict[str, Any]) -> bool:
        """Check if the response contains a final answer (no tool use)."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        # Check if there are any tool use requests
        for item in content:
            if "toolUse" in item:
                return False

        # If we have substantial text content and no tool use, it's likely a final answer
        for item in content:
            if "text" in item and len(item["text"]) > 200:
                return True

        return False

    def _extract_final_report(self, response: dict[str, Any]) -> str:
        """Extract the final report text from the response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        report_text = ""
        for item in content:
            if "text" in item:
                report_text += item["text"]

        return report_text

    def _process_tool_use(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Process tool use requests from the LLM response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        tool_results = []

        for item in content:
            if "toolUse" in item:
                tool_use = item["toolUse"]
                tool_name = tool_use.get("name")
                tool_input = tool_use.get("input", {})
                tool_use_id = tool_use.get("toolUseId")

                print(f"\n  → Tool: {tool_name}")
                print(f"    Input: {json.dumps(tool_input, indent=2)}")

                # Publish tool call start event
                self._publish_event(
                    "tool_call_start", {"tool": tool_name, "input": tool_input}
                )

                # Execute the tool
                result = self._execute_tool(tool_name, tool_input)

                print(f"    Result: {result.get('message', 'Success')}")

                # Publish tool call result event
                self._publish_event(
                    "tool_call_result", {"tool": tool_name, "result": result}
                )

                # Format tool result for Claude
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"json": result}],
                        }
                    }
                )

        return tool_results

    def _execute_tool(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool and log the execution."""
        self.execution_log.append({"tool": tool_name, "input": tool_input})

        result = self.tool_handler.handle_tool_call(tool_name, tool_input)

        self.execution_log[-1]["result"] = result

        # Publish collision detection events for specific tools
        if tool_name == "trace_impact_window" and result.get("success"):
            collision_detected = result.get("collision_detected", False)
            if collision_detected:
                self._publish_event(
                    "collision_detected", 
                    {"message": f"Result: Collision DETECTED - {result.get('message', '')}"}
                )
            else:
                self._publish_event(
                    "collision_detected", 
                    {"message": f"Result: Collision NOT DETECTED - {result.get('message', '')}"}
                )

        return result

    def _request_final_report(self) -> str:
        """Request a final report from the LLM."""
        print("\n[REQUESTING FINAL REPORT]")

        self.conversation_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": "Based on all the tool results, please provide your comprehensive accident analysis report now. Generate a report format that best suits the data you've analyzed, including all relevant findings, metrics, and conclusions. IMPORTANT: All distance measurements must be in miles and all speed measurements must be in miles per hour (MPH)."
                    }
                ],
            }
        )

        response = self._call_bedrock_with_tools()
        return self._extract_final_report(response)

    def _format_llm_report(self, report_text: str, iterations: int) -> dict[str, Any]:
        """Format the LLM-generated report into structured output."""
        return {
            "success": True,
            "report_type": "llm_generated",
            "model": self.current_model_id,
            "models_tried": self.config.bedrock_models[: self.current_model_index + 1],
            "model_attempts": self.model_attempts,
            "report": report_text,
            "iterations": iterations,
            "execution_log": self.execution_log,
            "conversation_history": self.conversation_history,
            "config": {
                "track_ids": self.config.track_ids,
                "frame_range": self.config.frame_range,
                "iou_threshold": self.config.iou_threshold,
                "distance_threshold_m": self.config.distance_threshold_m,
                "detections_file": self.config.detections_file,
            },
        }


def main():
    """Main entry point for LLM-powered accident analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-Powered Accident Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze collision between tracks 7 and 14
  python llm_agent.py --track-ids 7 14

  # Analyze specific frame range
  python llm_agent.py --track-ids 7 14 --frame-range 2 30

  # Use different AWS region and models
  python llm_agent.py --track-ids 7 14 --aws-region us-west-2 --bedrock-models anthropic.claude-3-5-sonnet-20241022-v2:0 anthropic.claude-3-5-haiku-20241022-v1:0

  # Adjust detection thresholds
  python llm_agent.py --track-ids 7 14 --iou-threshold 0.02 --distance-threshold 3.0

  # Save report to file
  python llm_agent.py --track-ids 7 14 --output report.json

  # Use custom detections file
  python llm_agent.py --track-ids 7 14 --detections-file /path/to/detections.jsonl
        """,
    )

    parser.add_argument(
        "--track-ids",
        type=int,
        nargs=2,
        required=True,
        metavar=("ID1", "ID2"),
        help="Two track IDs to analyze for collision (e.g., 7 14)",
    )

    parser.add_argument(
        "--frame-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Optional frame range to analyze (e.g., 2 30)",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.01,
        help="IoU threshold for collision detection (default: 0.01)",
    )

    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=5.0,
        help="Distance threshold in meters (default: 5.0)",
    )

    parser.add_argument(
        "--persistence-frames",
        type=int,
        default=3,
        help="Number of frames to persist overlap (default: 3)",
    )

    parser.add_argument(
        "--padding-frames",
        type=int,
        default=10,
        help="Number of frames to include before/after impact (default: 10)",
    )

    parser.add_argument(
        "--detections-file",
        type=str,
        default="../process_video/detections.jsonl",
        help="Path to detections.jsonl file (default: ../process_video/detections.jsonl)",
    )

    parser.add_argument(
        "--output", type=str, help="Output file for analysis report (JSON format)"
    )

    parser.add_argument(
        "--aws-region",
        type=str,
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)",
    )

    parser.add_argument(
        "--bedrock-models",
        type=str,
        nargs="+",
        default=[
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        ],
        help="List of Bedrock model IDs to try in order (default: multiple Claude models)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum iterations for LLM agent loop (default: 20)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (default: 0.0)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for LLM response (default: 4096)",
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.track_ids) != 2:
        parser.error("Exactly 2 track IDs must be provided")

    if args.frame_range and len(args.frame_range) != 2:
        parser.error("Frame range must have exactly 2 values (start and end)")

    if args.frame_range and args.frame_range[0] >= args.frame_range[1]:
        parser.error("Frame range start must be less than end")

    # Check if detections file exists
    detections_path = Path(args.detections_file)
    if not detections_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        detections_path = script_dir / args.detections_file
        if not detections_path.exists():
            parser.error(f"Detections file not found: {args.detections_file}")

    # Create agent configuration
    config = LLMAgentConfig(
        track_ids=args.track_ids,
        frame_range=tuple(args.frame_range) if args.frame_range else None,
        iou_threshold=args.iou_threshold,
        distance_threshold_m=args.distance_threshold,
        persistence_frames=args.persistence_frames,
        padding_frames=args.padding_frames,
        detections_file=str(detections_path),
        aws_region=args.aws_region,
        bedrock_models=args.bedrock_models,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Create and run agent
    try:
        agent = LLMAccidentAnalysisAgent(config)
        report = agent.analyze()

        if not report.get("success"):
            print("\n❌ Analysis failed!")
            if "error" in report:
                print(f"Error: {report['error']}")
            if "suggestion" in report:
                print(f"Suggestion: {report['suggestion']}")
            return 1

        # Save report if output file specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n✓ Report saved to: {args.output}")

        print("\n" + "=" * 80)
        print("✓ ANALYSIS COMPLETE")
        print("=" * 80)

        # Display the LLM-generated report
        print(f"\nLLM-Generated Report ({report['model']}):")
        print(f"Iterations: {report['iterations']}")
        if len(report["models_tried"]) > 1:
            print(f"Models tried: {', '.join(report['models_tried'])}")
            print(f"Model attempts: {report['model_attempts']}")
        print("\n" + "-" * 80)
        print(report["report"])
        print("-" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    exit_code = main()
    sys.exit(exit_code)

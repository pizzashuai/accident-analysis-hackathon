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

import boto3
from botocore.exceptions import ClientError

from ...config import settings
from .agent_tools import AgentToolHandler


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
    bedrock_model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    max_iterations: int = 20
    temperature: float = 0.0
    max_tokens: int = 4096


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

Your task is to analyze vehicle detection data and generate comprehensive accident analysis reports. You have access to specialized tools for data analysis.

CRITICAL REQUIREMENTS:
- Analyze the detection data thoroughly using available tools
- Generate reports that match the data characteristics and complexity
- Include specific frame citations, timestamps, and metrics in your analysis
- Adapt your report format based on what you find in the data
- Be precise about collision vs near-miss determinations
- Highlight data quality issues and assumptions

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
4. Build timeline of events with specific citations
5. Report data quality issues and assumptions
6. Generate a comprehensive analysis report

REPORT GENERATION:
- Let the data guide your report structure and format
- Include relevant sections based on what you find
- Use appropriate detail level based on data complexity
- Include specific metrics, frame numbers, and timestamps
- Highlight any data limitations or assumptions
- Provide clear conclusions with supporting evidence

Generate a professional, data-driven accident analysis report."""

    def __init__(self, config: LLMAgentConfig):
        """Initialize the LLM agent."""
        self.config = config
        self.tool_handler = AgentToolHandler(detections_file=config.detections_file)
        self.conversation_history: list[dict[str, Any]] = []
        self.execution_log: list[dict[str, Any]] = []

        # Initialize AWS Bedrock client
        self.bedrock_client = self._initialize_bedrock_client()

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
            return {
                "success": False,
                "error": "Bedrock client not available. Please configure AWS credentials.",
                "suggestion": "Run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables",
            }

        print("=" * 80)
        print("LLM-POWERED ACCIDENT ANALYSIS")
        print("=" * 80)
        print(f"Analyzing tracks: {self.config.track_ids}")
        print(f"Using model: {self.config.bedrock_model_id}")
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

            # Call Claude with tool use capability
            response = self._call_bedrock_with_tools()

            # Check if we got a final answer
            if self._is_final_answer(response):
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
            final_report = self._request_final_report()

        # Format and return the report
        return self._format_llm_report(final_report, iteration)

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

Start by loading the detection data for the specified tracks."""
        return query

    def _call_bedrock_with_tools(self) -> dict[str, Any]:
        """Call AWS Bedrock Claude with tool use capability."""
        if not self.bedrock_client:
            raise RuntimeError("Bedrock client not initialized")

        # Prepare tool definitions for Claude
        tools = self._prepare_tool_definitions()

        # Prepare messages
        messages = self.conversation_history.copy()

        # Call Bedrock
        try:
            response = self.bedrock_client.converse(
                modelId=self.config.bedrock_model_id,
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
            print(f"\n✗ Error calling Bedrock: {str(e)}")
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
                    "description": "Build a structured timeline of events from approach to separation. Returns timeline entries with frame, timestamp, metrics, and narrative.",
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

                # Execute the tool
                result = self._execute_tool(tool_name, tool_input)

                print(f"    Result: {result.get('message', 'Success')}")

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

        return result

    def _request_final_report(self) -> str:
        """Request a final report from the LLM."""
        print("\n[REQUESTING FINAL REPORT]")

        self.conversation_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": "Based on all the tool results, please provide your comprehensive accident analysis report now. Generate a report format that best suits the data you've analyzed, including all relevant findings, metrics, and conclusions."
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
            "model": self.config.bedrock_model_id,
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
  
  # Use different AWS region and model
  python llm_agent.py --track-ids 7 14 --aws-region us-west-2 --bedrock-model anthropic.claude-3-5-sonnet-20241022-v2:0
  
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
        "--bedrock-model",
        type=str,
        default="anthropic.claude-3-haiku-20240307-v1:0",
        help="Bedrock model ID (default: Claude 3.5 Sonnet v2)",
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
        bedrock_model_id=args.bedrock_model,
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

"""
Agent Core for Accident Analysis.
Implements the orchestration logic from agent-plan.md.
Uses AWS Bedrock Claude 4 for dynamic tool selection and report generation.
"""

import json
from dataclasses import dataclass
from typing import Any

import boto3

from ...config import settings
from .agent_tools import AgentToolHandler
from .dynamic_agent_core import DynamicAgentCore


@dataclass
class AgentConfig:
    """Configuration for the accident analysis agent."""

    track_ids: list[int]
    frame_range: tuple | None = None
    iou_threshold: float = 0.01
    distance_threshold_m: float = 5.0
    persistence_frames: int = 3
    padding_frames: int = 10
    detections_file: str = "backend/src/common/features/process_video/detections.jsonl"
    use_dynamic_analysis: bool = False
    include_speed_analysis: bool = True
    include_heading_analysis: bool = False
    custom_timeline_stages: list[str] | None = None
    fps_hint: float | None = None
    vehicle_length_m: float = 4.5
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    use_llm_agent: bool = True
    max_iterations: int = 20


class AccidentAnalysisAgent:
    """
    Main agent for accident analysis following the blueprint from agent-plan.md.

    This agent orchestrates the analysis workflow:
    1. Load detections for specified track IDs
    2. Compute pair metrics (IoU, distances, speeds)
    3. Trace impact window to identify collision
    4. Build event timeline
    5. Report assumptions and data quality issues

    Supports both static (hardcoded) and dynamic (intelligent) tool selection.
    """

    SYSTEM_PROMPT = """You are an accident reconstruction analyst specializing in vehicle collision analysis.

Your goal is to analyze detection data from video processing to determine if a collision occurred and produce a comprehensive timeline and detailed report.

CRITICAL REQUIREMENTS:
- Every event citation must include frame number, timestamp, and key numeric metrics (IoU, distance, speed)
- Verify units: distances in meters, speeds in m/s or mph, time in seconds
- Use the available tools to gather data and make informed decisions
- Surface any missing metadata as warnings
- Distinguish between actual collisions and near-miss events
- Generate a complete, professional accident analysis report

AVAILABLE TOOLS:
1. load_detections - Load paired detection records for specified track IDs
2. compute_pair_metrics - Calculate IoU, distances, and speeds between vehicles
3. trace_impact_window - Identify collision/near-miss frames
4. build_timeline - Generate structured timeline of events
5. report_assumptions - Identify data quality issues

ANALYSIS WORKFLOW:
1. Load detections with the specified track IDs using load_detections
2. Compute pair metrics using compute_pair_metrics to get IoU, distances, and speeds
3. Trace impact window using trace_impact_window to identify collision or near-miss
4. Build timeline using build_timeline to get structured event sequence
5. Report assumptions using report_assumptions to identify data quality issues
6. Synthesize all data into a comprehensive report

FINAL REPORT FORMAT:
Your final response should be a complete accident analysis report with:

1. **EXECUTIVE SUMMARY**: Brief overview of collision/near-miss determination
2. **EVENT TIMELINE**: Chronological sequence of events with citations (frame, timestamp, IoU, distance, speed)
3. **IMPACT ANALYSIS**: Detailed analysis of the collision or near-miss with severity indicators
4. **DATA QUALITY**: Assumptions, limitations, and data quality issues
5. **CONCLUSION**: Final determination with supporting evidence

Use the tools to gather all necessary data, then synthesize it into a professional report."""

    def __init__(self, config: AgentConfig):
        """
        Initialize the agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.tool_handler = AgentToolHandler(detections_file=config.detections_file)
        self.execution_log: list[dict[str, Any]] = []
        self.conversation_history: list[dict[str, Any]] = []

        # Initialize AWS Bedrock client if using LLM agent
        self.bedrock_client: Any | None = None
        if config.use_llm_agent:
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )

        # Initialize dynamic agent if requested
        if config.use_dynamic_analysis:
            self.dynamic_agent = DynamicAgentCore(self._config_to_dict())
        else:
            self.dynamic_agent = None

    def analyze(self) -> dict[str, Any]:
        """
        Execute the full accident analysis workflow.

        Returns:
            Complete analysis results with timeline, impact summary, and assumptions
        """
        # Use LLM-powered analysis if configured
        if self.config.use_llm_agent and self.bedrock_client:
            return self._analyze_with_llm()

        # Use dynamic analysis if configured
        if self.config.use_dynamic_analysis and self.dynamic_agent:
            return self.dynamic_agent.analyze()

        # Fall back to static analysis
        return self._analyze_static()

    def _config_to_dict(self) -> dict[str, Any]:
        """Convert AgentConfig to dictionary for dynamic agent."""
        return {
            "track_ids": self.config.track_ids,
            "frame_range": self.config.frame_range,
            "iou_threshold": self.config.iou_threshold,
            "distance_threshold_m": self.config.distance_threshold_m,
            "persistence_frames": self.config.persistence_frames,
            "padding_frames": self.config.padding_frames,
            "detections_file": self.config.detections_file,
            "include_speed_analysis": self.config.include_speed_analysis,
            "include_heading_analysis": self.config.include_heading_analysis,
            "custom_timeline_stages": self.config.custom_timeline_stages,
            "fps_hint": self.config.fps_hint,
            "vehicle_length_m": self.config.vehicle_length_m,
        }

    def _analyze_static(self) -> dict[str, Any]:
        """
        Execute the static (hardcoded) accident analysis workflow.

        Returns:
            Complete analysis results with timeline, impact summary, and assumptions
        """
        print("=" * 80)
        print("ACCIDENT RECONSTRUCTION ANALYSIS (STATIC)")
        print("=" * 80)
        print(f"\nAnalyzing collision between tracks: {self.config.track_ids}")
        print(
            f"Frame range: {self.config.frame_range if self.config.frame_range else 'All frames'}"
        )
        print(
            f"Detection threshold - IoU: {self.config.iou_threshold}, Distance: {self.config.distance_threshold_m}m"
        )
        print("\n" + "-" * 80)

        # Step 1: Load detections
        print("\n[STEP 1] Loading detection data...")
        load_result = self._execute_tool(
            "load_detections",
            {
                "track_ids": self.config.track_ids,
                "frame_range": list(self.config.frame_range)
                if self.config.frame_range
                else None,
                "require_pairing": True,
            },
        )

        if not load_result.get("success"):
            return self._format_error("Failed to load detections", load_result)

        print(f"✓ Loaded {load_result['total_frames']} frames")
        print(f"  Frame range: {load_result['frame_range']}")
        if load_result["time_range"]:
            print(
                f"  Time span: {load_result['time_range'][0]:.3f}s - {load_result['time_range'][1]:.3f}s"
            )
        else:
            print("  Time span: Not available")
        print(
            f"  Estimated FPS: {load_result['fps_estimated']:.1f}"
            if load_result["fps_estimated"]
            else "  FPS: Not available"
        )

        if load_result["missing_frames"] > 0:
            print(
                f"  ⚠ Warning: {load_result['missing_frames']} missing frames detected"
            )

        # Step 2: Compute metrics
        print("\n[STEP 2] Computing collision metrics...")
        metrics_result = self._execute_tool(
            "compute_pair_metrics",
            {"iou_threshold": self.config.iou_threshold, "include_headings": False},
        )

        if not metrics_result.get("success"):
            return self._format_error("Failed to compute metrics", metrics_result)

        print(f"✓ Computed metrics for {metrics_result['total_frames']} frames")
        print(
            f"  IoU range: {metrics_result['min_iou']:.3f} - {metrics_result['max_iou']:.3f} (avg: {metrics_result['avg_iou']:.3f})"
        )
        print(
            f"  Minimum distance: {metrics_result['min_distance_m']:.1f}m"
            if metrics_result["min_distance_m"]
            else "  Distance data: Not available"
        )
        print(
            f"  Collision candidates: {metrics_result['collision_candidates']} frames"
        )

        # Step 3: Trace impact window
        print("\n[STEP 3] Tracing impact window...")
        impact_result = self._execute_tool(
            "trace_impact_window",
            {
                "iou_threshold": self.config.iou_threshold,
                "distance_threshold_m": self.config.distance_threshold_m,
                "persistence_frames": self.config.persistence_frames,
            },
        )

        if not impact_result.get("success"):
            return self._format_error("Failed to trace impact window", impact_result)

        collision_detected = impact_result["collision_detected"]
        print(
            f"✓ Impact analysis complete: {'COLLISION DETECTED' if collision_detected else 'NO COLLISION (Near-miss)'}"
        )

        if collision_detected:
            print(f"  First contact: Frame {impact_result['first_contact_frame']}")
            print(f"  Last overlap: Frame {impact_result['last_overlap_frame']}")
            print(
                f"  Impact duration: {impact_result['overlap_duration_frames']} frames"
            )
            print(f"  Impact frames analyzed: {impact_result['impact_frames_count']}")
        else:
            print(
                f"  Closest approach: {impact_result['closest_approach_distance_m']:.1f}m at frame {impact_result['closest_approach_frame']}"
            )

        print("\n  Diagnostic notes:")
        for note in impact_result["diagnostic_notes"]:
            print(f"    - {note}")

        # Step 4: Build timeline
        print("\n[STEP 4] Building event timeline...")
        timeline_result = self._execute_tool(
            "build_timeline", {"padding_frames": self.config.padding_frames}
        )

        if not timeline_result.get("success"):
            return self._format_error("Failed to build timeline", timeline_result)

        print(f"✓ Timeline built with {len(timeline_result['timeline'])} key stages")

        # Step 5: Report assumptions
        print("\n[STEP 5] Analyzing data quality...")
        assumptions_result = self._execute_tool(
            "report_assumptions", {"warn_if_missing": ["world_coords", "speed_mph"]}
        )

        if not assumptions_result.get("success"):
            return self._format_error(
                "Failed to report assumptions", assumptions_result
            )

        print(f"✓ Identified {assumptions_result['total_issues']} data quality issues")

        # Step 6: Generate final report
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE - GENERATING REPORT")
        print("=" * 80)

        final_report = self._generate_final_report(
            load_result=load_result,
            metrics_result=metrics_result,
            impact_result=impact_result,
            timeline_result=timeline_result,
            assumptions_result=assumptions_result,
        )

        return final_report

    def _execute_tool(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute a tool and log the execution.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool execution result
        """
        self.execution_log.append({"tool": tool_name, "input": tool_input})

        result = self.tool_handler.handle_tool_call(tool_name, tool_input)

        self.execution_log[-1]["result"] = result

        return result

    def _format_error(
        self, message: str, error_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Format an error response."""
        return {
            "success": False,
            "error": message,
            "details": error_result,
            "execution_log": self.execution_log,
        }

    def _generate_final_report(
        self,
        load_result: dict[str, Any],
        metrics_result: dict[str, Any],
        impact_result: dict[str, Any],
        timeline_result: dict[str, Any],
        assumptions_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate the final analysis report.

        Args:
            load_result: Results from load_detections
            metrics_result: Results from compute_pair_metrics
            impact_result: Results from trace_impact_window
            timeline_result: Results from build_timeline
            assumptions_result: Results from report_assumptions

        Returns:
            Complete analysis report
        """
        collision_detected = impact_result["collision_detected"]

        # Generate narrative summary
        narrative = self._generate_narrative(
            collision_detected=collision_detected,
            impact_result=impact_result,
            timeline_result=timeline_result,
            load_result=load_result,
        )

        # Format timeline with citations
        formatted_timeline = self._format_timeline(timeline_result["timeline"])

        # Generate impact summary
        impact_summary = self._generate_impact_summary(
            collision_detected=collision_detected,
            impact_result=impact_result,
            metrics_result=metrics_result,
        )

        report = {
            "success": True,
            "collision_detected": collision_detected,
            "narrative_summary": narrative,
            "timeline": formatted_timeline,
            "summary_table": timeline_result["summary_table"],
            "impact_analysis": impact_summary,
            "data_quality": {
                "total_frames_analyzed": load_result["total_frames"],
                "frame_range": load_result["frame_range"],
                "time_range": load_result["time_range"],
                "fps_estimated": load_result.get("fps_estimated"),
                "missing_frames": load_result["missing_frames"],
                "assumptions": assumptions_result["assumptions"],
            },
            "metrics_summary": {
                "max_iou": metrics_result["max_iou"],
                "min_distance_m": metrics_result["min_distance_m"],
                "collision_candidates": metrics_result["collision_candidates"],
            },
            "execution_log": self.execution_log,
        }

        # Print the report
        self._print_report(report)

        return report

    def _generate_narrative(
        self,
        collision_detected: bool,
        impact_result: dict[str, Any],
        timeline_result: dict[str, Any],
        load_result: dict[str, Any],
    ) -> str:
        """Generate narrative summary of the incident."""
        track_ids = self.config.track_ids
        time_range = load_result.get("time_range")
        duration = (
            time_range[1] - time_range[0] if time_range and len(time_range) == 2 else 0
        )

        if collision_detected:
            first_frame = impact_result["first_contact_frame"]
            last_frame = impact_result["last_overlap_frame"]
            impact_duration = impact_result["overlap_duration_frames"]

            narrative = (
                f"COLLISION DETECTED between vehicles (Track {track_ids[0]} and Track {track_ids[1]}) "
                f"over a {duration:.2f}-second observation period. "
                f"Initial contact occurred at frame {first_frame}, with overlap persisting for "
                f"{impact_duration} frames until frame {last_frame}. "
                f"The collision involved {impact_result['impact_frames_count']} frames meeting "
                f"the detection criteria (IoU > {self.config.iou_threshold} or distance < {self.config.distance_threshold_m}m)."
            )
        else:
            closest_frame = impact_result["closest_approach_frame"]
            closest_distance = impact_result["closest_approach_distance_m"]

            narrative = (
                f"NEAR-MISS EVENT between vehicles (Track {track_ids[0]} and Track {track_ids[1]}) "
                f"over a {duration:.2f}-second observation period. "
                f"No collision was detected based on the analysis criteria. "
                f"The vehicles achieved their closest approach of {closest_distance:.1f}m at frame {closest_frame}. "
                f"While the vehicles came close, they did not meet the collision threshold "
                f"(IoU > {self.config.iou_threshold} or distance < {self.config.distance_threshold_m}m for "
                f"{self.config.persistence_frames}+ consecutive frames)."
            )

        return narrative

    def _format_timeline(self, timeline: list[dict[str, Any]]) -> list[str]:
        """Format timeline entries with full citations."""
        formatted = []

        for event in timeline:
            stage = event["stage"]
            frame = event["frame"]
            timestamp = event["timestamp"]
            metrics = event["metrics"]
            narrative = event["narrative"]

            citation = (
                f"[{stage.upper()}] Frame {frame} (t={timestamp:.3f}s): "
                f"{narrative} | "
                f"IoU={metrics['iou']:.3f}, "
                f"Distance={metrics['world_distance_m']:.1f}m"
                if metrics["world_distance_m"]
                else "Distance=N/A"
            )

            if metrics.get("relative_speed_mps"):
                citation += f", Speed_diff={metrics['relative_speed_mps']:.1f}m/s"

            formatted.append(citation)

        return formatted

    def _generate_impact_summary(
        self,
        collision_detected: bool,
        impact_result: dict[str, Any],
        metrics_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate detailed impact analysis summary."""
        if collision_detected:
            return {
                "type": "COLLISION",
                "severity_indicators": {
                    "max_iou": metrics_result["max_iou"],
                    "min_distance_m": metrics_result["min_distance_m"],
                    "impact_duration_frames": impact_result["overlap_duration_frames"],
                    "impact_frames_count": impact_result["impact_frames_count"],
                },
                "key_frames": {
                    "first_contact": impact_result["first_contact_frame"],
                    "last_overlap": impact_result["last_overlap_frame"],
                    "closest_approach": impact_result["closest_approach_frame"],
                },
                "diagnostic_notes": impact_result["diagnostic_notes"],
            }
        else:
            return {
                "type": "NEAR-MISS",
                "closest_approach": {
                    "frame": impact_result["closest_approach_frame"],
                    "distance_m": impact_result["closest_approach_distance_m"],
                },
                "max_iou": metrics_result["max_iou"],
                "diagnostic_notes": impact_result["diagnostic_notes"],
            }

    def _print_report(self, report: dict[str, Any]):
        """Print the analysis report to console."""
        print("\n" + "=" * 80)
        print("FINAL ANALYSIS REPORT")
        print("=" * 80)

        # Narrative Summary
        print("\n### SCENARIO SUMMARY ###")
        print(report["narrative_summary"])

        # Timeline
        print("\n### EVENT TIMELINE ###")
        for i, event in enumerate(report["timeline"], 1):
            print(f"{i}. {event}")

        # Impact Analysis
        print("\n### IMPACT ANALYSIS ###")
        impact = report["impact_analysis"]
        print(f"Event Type: {impact['type']}")

        if impact["type"] == "COLLISION":
            print("\nSeverity Indicators:")
            print(f"  - Maximum IoU: {impact['severity_indicators']['max_iou']:.3f}")
            print(
                f"  - Minimum distance: {impact['severity_indicators']['min_distance_m']:.1f}m"
                if impact["severity_indicators"]["min_distance_m"]
                else "  - Distance data: Not available"
            )
            print(
                f"  - Impact duration: {impact['severity_indicators']['impact_duration_frames']} frames"
            )
            print(
                f"  - Impact frames: {impact['severity_indicators']['impact_frames_count']} total"
            )

            print("\nKey Frames:")
            print(f"  - First contact: {impact['key_frames']['first_contact']}")
            print(f"  - Last overlap: {impact['key_frames']['last_overlap']}")
            print(f"  - Closest approach: {impact['key_frames']['closest_approach']}")
        else:
            print("\nClosest Approach:")
            print(f"  - Frame: {impact['closest_approach']['frame']}")
            print(f"  - Distance: {impact['closest_approach']['distance_m']:.1f}m")
            print(f"  - Maximum IoU: {impact['max_iou']:.3f}")

        print("\nDiagnostic Notes:")
        for note in impact["diagnostic_notes"]:
            print(f"  - {note}")

        # Summary Table
        print("\n### SUMMARY TABLE ###")
        if report["summary_table"]:
            headers = list(report["summary_table"][0].keys())
            print("  " + " | ".join(headers))
            print("  " + "-" * (sum(len(h) for h in headers) + 3 * len(headers)))
            for row in report["summary_table"]:
                print("  " + " | ".join(str(row[h]) for h in headers))

        # Data Quality
        print("\n### DATA QUALITY & ASSUMPTIONS ###")
        quality = report["data_quality"]
        print(f"Total frames analyzed: {quality['total_frames_analyzed']}")
        print(f"Frame range: {quality['frame_range']}")
        if quality.get("time_range"):
            print(
                f"Time span: {quality['time_range'][0]:.3f}s - {quality['time_range'][1]:.3f}s"
            )
        else:
            print("Time span: Not available")
        print(
            f"Estimated FPS: {quality['fps_estimated']:.1f}"
            if quality["fps_estimated"]
            else "FPS: Not available"
        )

        if quality["missing_frames"] > 0:
            print(f"\n⚠ Data Gaps: {quality['missing_frames']} missing frames")

        if quality["assumptions"]:
            print("\nAssumptions & Warnings:")
            for assumption in quality["assumptions"]:
                print(f"  ⚠ {assumption}")

        print("\n" + "=" * 80)

    def _analyze_with_llm(self) -> dict[str, Any]:
        """
        Execute LLM-powered accident analysis using AWS Bedrock Claude 4.

        The LLM decides which tools to call and generates the final report.

        Returns:
            Complete analysis results with LLM-generated timeline and report
        """
        print("=" * 80)
        print("ACCIDENT RECONSTRUCTION ANALYSIS (LLM-POWERED)")
        print("=" * 80)
        print(f"\nAnalyzing collision between tracks: {self.config.track_ids}")
        print(f"Using AWS Bedrock model: {self.config.bedrock_model_id}")
        print("\n" + "-" * 80)

        # Initialize conversation with user query
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
            # Request final report
            final_report = self._request_final_report()

        # Format and return the report
        return self._format_llm_report(final_report)

    def _create_initial_query(self) -> str:
        """Create the initial query for the LLM."""
        query = f"""Analyze the vehicle collision data for tracks {self.config.track_ids}.

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
3. Generate a comprehensive event timeline with specific frame citations
4. Provide a detailed accident analysis report

Start by loading the detection data for the specified tracks."""
        return query

    def _call_bedrock_with_tools(self) -> dict[str, Any]:
        """Call AWS Bedrock Claude with tool use capability."""
        if not self.bedrock_client:
            raise RuntimeError(
                "Bedrock client not initialized. Set use_llm_agent=True in config."
            )

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
                    "maxTokens": 4096,
                    "temperature": 0.0,
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

        # If we have text content and no tool use, it's likely a final answer
        for item in content:
            if "text" in item and len(item["text"]) > 100:
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

    def _request_final_report(self) -> str:
        """Request a final report from the LLM."""
        print("\n[REQUESTING FINAL REPORT]")

        self.conversation_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": "Based on all the tool results, please provide your comprehensive accident analysis report now. Include the executive summary, event timeline, impact analysis, data quality assessment, and conclusion."
                    }
                ],
            }
        )

        response = self._call_bedrock_with_tools()
        return self._extract_final_report(response)

    def _format_llm_report(self, report_text: str) -> dict[str, Any]:
        """Format the LLM-generated report into structured output."""
        return {
            "success": True,
            "report_type": "llm_generated",
            "model": self.config.bedrock_model_id,
            "report": report_text,
            "execution_log": self.execution_log,
            "conversation_history": self.conversation_history,
            "iterations": len(
                [msg for msg in self.conversation_history if msg["role"] == "assistant"]
            ),
        }

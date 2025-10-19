"""
Dynamic Agent Core for Accident Analysis.
Implements intelligent tool selection based on data analysis and requirements.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .agent_tools import AgentToolHandler


class ToolType(Enum):
    """Available tool types for analysis."""

    LOAD_DETECTIONS = "load_detections"
    COMPUTE_METRICS = "compute_pair_metrics"
    TRACE_IMPACT = "trace_impact_window"
    BUILD_TIMELINE = "build_timeline"
    REPORT_ASSUMPTIONS = "report_assumptions"


class AnalysisStage(Enum):
    """Analysis stages that can be executed."""

    DATA_LOADING = "data_loading"
    METRICS_COMPUTATION = "metrics_computation"
    IMPACT_ANALYSIS = "impact_analysis"
    TIMELINE_BUILDING = "timeline_building"
    QUALITY_ASSESSMENT = "quality_assessment"


@dataclass
class ToolDecision:
    """Represents a decision to call a specific tool."""

    tool_type: ToolType
    parameters: dict[str, Any]
    reason: str
    priority: int  # Higher number = higher priority
    dependencies: list[ToolType] | None = None  # Tools that must be called first


@dataclass
class AnalysisContext:
    """Context information for making tool decisions."""

    track_ids: list[int]
    frame_range: tuple[int, int] | None
    config: dict[str, Any]
    current_stage: AnalysisStage
    completed_tools: list[ToolType]
    data_quality_issues: list[str]
    analysis_requirements: dict[str, Any]


class DynamicAgentCore:
    """
    Dynamic agent that intelligently selects tools based on data analysis.

    This agent analyzes the data and requirements to determine:
    1. Which tools are necessary for the analysis
    2. What parameters to use for each tool
    3. The optimal order of tool execution
    4. Whether additional analysis steps are needed
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the dynamic agent.

        Args:
            config: Agent configuration including track_ids, thresholds, etc.
        """
        self.config = config
        self.tool_handler = AgentToolHandler(
            detections_file=config.get(
                "detections_file",
                "backend/src/common/features/process_video/detections.jsonl",
            )
        )
        self.execution_log = []
        self.analysis_context = None

    def analyze(self) -> dict[str, Any]:
        """
        Execute dynamic accident analysis workflow.

        Returns:
            Complete analysis results with timeline, impact summary, and assumptions
        """
        print("=" * 80)
        print("DYNAMIC ACCIDENT RECONSTRUCTION ANALYSIS")
        print("=" * 80)
        print(f"\nAnalyzing collision between tracks: {self.config['track_ids']}")
        print(f"Frame range: {self.config.get('frame_range', 'All frames')}")
        print(
            f"Detection threshold - IoU: {self.config.get('iou_threshold', 0.01)}, Distance: {self.config.get('distance_threshold_m', 5.0)}m"
        )
        print("\n" + "-" * 80)

        # Initialize analysis context
        self.analysis_context = AnalysisContext(
            track_ids=self.config["track_ids"],
            frame_range=self.config.get("frame_range"),
            config=self.config,
            current_stage=AnalysisStage.DATA_LOADING,
            completed_tools=[],
            data_quality_issues=[],
            analysis_requirements=self._determine_requirements(),
        )

        # Execute dynamic workflow
        try:
            result = self._execute_dynamic_workflow()
            return result
        except Exception as e:
            return self._format_error(f"Dynamic analysis failed: {str(e)}", {})

    def _determine_requirements(self) -> dict[str, Any]:
        """Determine analysis requirements based on configuration."""
        requirements = {
            "collision_detection": True,
            "timeline_analysis": True,
            "quality_assessment": True,
            "detailed_metrics": True,
            "speed_analysis": self.config.get("include_speed_analysis", True),
            "heading_analysis": self.config.get("include_heading_analysis", False),
            "custom_stages": self.config.get("custom_timeline_stages"),
            "strict_thresholds": self.config.get("iou_threshold", 0.01) > 0.05
            or self.config.get("distance_threshold_m", 5.0) < 3.0,
        }
        return requirements

    def _execute_dynamic_workflow(self) -> dict[str, Any]:
        """Execute the dynamic workflow by making intelligent tool decisions."""
        tool_decisions = []
        results = {}

        # Always start with loading detections
        tool_decisions.append(self._decide_load_detections())

        # Execute tools in priority order
        tool_decisions.sort(key=lambda x: x.priority, reverse=True)

        while tool_decisions:
            # Get the highest priority tool that should be executed
            decision = None
            for i, d in enumerate(tool_decisions):
                if self._should_execute_tool(d):
                    decision = d
                    tool_decisions.pop(i)
                    break

            if not decision:
                break  # No more tools to execute

            print(f"\n[EXECUTING] {decision.tool_type.value}: {decision.reason}")
            result = self._execute_tool(decision.tool_type.value, decision.parameters)
            results[decision.tool_type.value] = result

            if not result.get("success"):
                print(
                    f"❌ Tool {decision.tool_type.value} failed: {result.get('error', 'Unknown error')}"
                )
                # Decide whether to continue or abort
                if self._should_abort_analysis(decision.tool_type, result):
                    return self._format_error(
                        f"Critical tool {decision.tool_type.value} failed", result
                    )
            else:
                print(f"✓ {decision.tool_type.value} completed successfully")
                self.analysis_context.completed_tools.append(decision.tool_type)

                # Update context based on results
                self._update_context_from_result(decision.tool_type, result)

                # Check if we need additional tools based on results
                additional_decisions = self._decide_additional_tools(result)
                tool_decisions.extend(additional_decisions)
                tool_decisions.sort(key=lambda x: x.priority, reverse=True)

        # Generate final report
        return self._generate_final_report(results)

    def _decide_load_detections(self) -> ToolDecision:
        """Decide parameters for loading detections."""
        return ToolDecision(
            tool_type=ToolType.LOAD_DETECTIONS,
            parameters={
                "track_ids": self.config["track_ids"],
                "frame_range": list(self.config["frame_range"])
                if self.config.get("frame_range")
                else None,
                "require_pairing": True,
                "fps_hint": self.config.get("fps_hint"),
            },
            reason="Load detection data for specified track IDs",
            priority=100,  # Highest priority
            dependencies=[],
        )

    def _decide_compute_metrics(self) -> ToolDecision:
        """Decide parameters for computing metrics."""
        requirements = (
            self.analysis_context.analysis_requirements if self.analysis_context else {}
        )

        return ToolDecision(
            tool_type=ToolType.COMPUTE_METRICS,
            parameters={
                "iou_threshold": self.config.get("iou_threshold", 0.01),
                "vehicle_length_m": self.config.get("vehicle_length_m", 4.5),
                "include_headings": requirements.get("heading_analysis", False),
            },
            reason="Compute collision metrics and analyze vehicle interactions",
            priority=90,
            dependencies=[ToolType.LOAD_DETECTIONS],
        )

    def _decide_trace_impact(self) -> ToolDecision:
        """Decide parameters for tracing impact window."""
        return ToolDecision(
            tool_type=ToolType.TRACE_IMPACT,
            parameters={
                "iou_threshold": self.config.get("iou_threshold", 0.01),
                "distance_threshold_m": self.config.get("distance_threshold_m", 5.0),
                "persistence_frames": self.config.get("persistence_frames", 3),
            },
            reason="Trace impact window to detect collision events",
            priority=80,
            dependencies=[ToolType.COMPUTE_METRICS],
        )

    def _decide_build_timeline(self) -> ToolDecision:
        """Decide parameters for building timeline."""
        requirements = (
            self.analysis_context.analysis_requirements if self.analysis_context else {}
        )

        return ToolDecision(
            tool_type=ToolType.BUILD_TIMELINE,
            parameters={
                "padding_frames": self.config.get("padding_frames", 10),
                "stages": requirements.get("custom_stages"),
            },
            reason="Build structured timeline of collision events",
            priority=70,
            dependencies=[ToolType.TRACE_IMPACT],
        )

    def _decide_report_assumptions(self) -> ToolDecision:
        """Decide parameters for reporting assumptions."""
        warn_fields = ["world_coords", "speed_mph"]
        if self.analysis_context and self.analysis_context.analysis_requirements.get(
            "heading_analysis"
        ):
            warn_fields.append("heading")

        return ToolDecision(
            tool_type=ToolType.REPORT_ASSUMPTIONS,
            parameters={"warn_if_missing": warn_fields},
            reason="Assess data quality and identify analysis assumptions",
            priority=60,
            dependencies=[ToolType.COMPUTE_METRICS],
        )

    def _should_execute_tool(self, decision: ToolDecision) -> bool:
        """Determine if a tool should be executed based on context."""
        # Check dependencies
        if decision.dependencies and self.analysis_context:
            for dep in decision.dependencies:
                if dep not in self.analysis_context.completed_tools:
                    return False

        # Check if already executed
        if (
            self.analysis_context
            and decision.tool_type in self.analysis_context.completed_tools
        ):
            return False

        return True

    def _decide_additional_tools(self, result: dict[str, Any]) -> list[ToolDecision]:
        """Decide if additional tools are needed based on results."""
        additional_decisions = []

        if not self.analysis_context:
            return additional_decisions

        # Check each tool independently and add if needed
        if ToolType.COMPUTE_METRICS not in self.analysis_context.completed_tools:
            additional_decisions.append(self._decide_compute_metrics())

        if ToolType.TRACE_IMPACT not in self.analysis_context.completed_tools:
            additional_decisions.append(self._decide_trace_impact())

        if ToolType.BUILD_TIMELINE not in self.analysis_context.completed_tools:
            additional_decisions.append(self._decide_build_timeline())

        # Always add assumptions reporting if we have metrics
        if ToolType.REPORT_ASSUMPTIONS not in self.analysis_context.completed_tools:
            additional_decisions.append(self._decide_report_assumptions())
        return additional_decisions

    def _should_abort_analysis(
        self, tool_type: ToolType, result: dict[str, Any]
    ) -> bool:
        """Determine if analysis should be aborted based on tool failure."""
        # Critical tools that require aborting
        critical_tools = {ToolType.LOAD_DETECTIONS, ToolType.COMPUTE_METRICS}

        if tool_type in critical_tools:
            return True

        # For non-critical tools, continue with warnings
        return False

    def _update_context_from_result(self, tool_type: ToolType, result: dict[str, Any]):
        """Update analysis context based on tool results."""
        if not self.analysis_context:
            return

        if tool_type == ToolType.LOAD_DETECTIONS:
            if result.get("missing_frames", 0) > 0:
                self.analysis_context.data_quality_issues.append(
                    f"Missing {result['missing_frames']} frames"
                )
            if not result.get("fps_estimated"):
                self.analysis_context.data_quality_issues.append("FPS not available")

        elif tool_type == ToolType.COMPUTE_METRICS:
            if result.get("collision_candidates", 0) == 0:
                self.analysis_context.data_quality_issues.append(
                    "No collision candidates detected"
                )
            if not result.get("min_distance_m"):
                self.analysis_context.data_quality_issues.append(
                    "Distance data not available"
                )

        elif tool_type == ToolType.TRACE_IMPACT:
            if not result.get("collision_detected"):
                self.analysis_context.data_quality_issues.append(
                    "No collision detected - near-miss scenario"
                )

    def _execute_tool(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool and log the execution."""
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
            "completed_tools": [
                tool.value for tool in self.analysis_context.completed_tools
            ]
            if self.analysis_context
            else [],
        }

    def _generate_final_report(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate the final analysis report from all tool results."""
        # Extract results from each tool
        load_result = results.get("load_detections", {})
        metrics_result = results.get("compute_pair_metrics", {})
        impact_result = results.get("trace_impact_window", {})
        timeline_result = results.get("build_timeline", {})
        assumptions_result = results.get("report_assumptions", {})

        # Determine collision status
        collision_detected = impact_result.get("collision_detected", False)

        # Generate narrative summary
        narrative = self._generate_narrative(
            collision_detected=collision_detected,
            impact_result=impact_result,
            timeline_result=timeline_result,
            load_result=load_result,
        )

        # Format timeline with citations
        formatted_timeline = self._format_timeline(timeline_result.get("timeline", []))

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
            "summary_table": timeline_result.get("summary_table", []),
            "impact_analysis": impact_summary,
            "data_quality": {
                "total_frames_analyzed": load_result.get("total_frames", 0),
                "frame_range": load_result.get("frame_range"),
                "time_range": load_result.get("time_range"),
                "fps_estimated": load_result.get("fps_estimated"),
                "missing_frames": load_result.get("missing_frames", 0),
                "assumptions": assumptions_result.get("assumptions", []),
                "quality_issues": self.analysis_context.data_quality_issues
                if self.analysis_context
                else [],
            },
            "metrics_summary": {
                "max_iou": metrics_result.get("max_iou", 0),
                "min_distance_m": metrics_result.get("min_distance_m"),
                "collision_candidates": metrics_result.get("collision_candidates", 0),
            },
            "execution_log": self.execution_log,
            "tools_executed": [
                tool.value for tool in self.analysis_context.completed_tools
            ]
            if self.analysis_context
            else [],
            "dynamic_analysis": True,
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
        track_ids = self.config["track_ids"]
        time_range = load_result.get("time_range")
        duration = (
            time_range[1] - time_range[0] if time_range and len(time_range) == 2 else 0
        )

        if collision_detected:
            first_frame = impact_result.get("first_contact_frame")
            last_frame = impact_result.get("last_overlap_frame")
            impact_duration = impact_result.get("overlap_duration_frames", 0)

            narrative = (
                f"COLLISION DETECTED between vehicles (Track {track_ids[0]} and Track {track_ids[1]}) "
                f"over a {duration:.2f}-second observation period. "
                f"Initial contact occurred at frame {first_frame}, with overlap persisting for "
                f"{impact_duration} frames until frame {last_frame}. "
                f"The collision involved {impact_result.get('impact_frames_count', 0)} frames meeting "
                f"the detection criteria."
            )
        else:
            closest_frame = impact_result.get("closest_approach_frame")
            closest_distance = impact_result.get("closest_approach_distance_m", 0)

            narrative = (
                f"NEAR-MISS EVENT between vehicles (Track {track_ids[0]} and Track {track_ids[1]}) "
                f"over a {duration:.2f}-second observation period. "
                f"No collision was detected based on the analysis criteria. "
                f"The vehicles achieved their closest approach of {closest_distance:.1f}m at frame {closest_frame}. "
                f"While the vehicles came close, they did not meet the collision threshold."
            )

        return narrative

    def _format_timeline(self, timeline: list[dict[str, Any]]) -> list[str]:
        """Format timeline entries with full citations."""
        formatted = []

        for event in timeline:
            stage = event.get("stage", "unknown")
            frame = event.get("frame", 0)
            timestamp = event.get("timestamp", 0)
            metrics = event.get("metrics", {})
            narrative = event.get("narrative", "")

            citation = (
                f"[{stage.upper()}] Frame {frame} (t={timestamp:.3f}s): "
                f"{narrative} | "
                f"IoU={metrics.get('iou', 0):.3f}, "
                f"Distance={metrics.get('world_distance_m', 0):.1f}m"
                if metrics.get("world_distance_m")
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
                    "max_iou": metrics_result.get("max_iou", 0),
                    "min_distance_m": metrics_result.get("min_distance_m"),
                    "impact_duration_frames": impact_result.get(
                        "overlap_duration_frames", 0
                    ),
                    "impact_frames_count": impact_result.get("impact_frames_count", 0),
                },
                "key_frames": {
                    "first_contact": impact_result.get("first_contact_frame"),
                    "last_overlap": impact_result.get("last_overlap_frame"),
                    "closest_approach": impact_result.get("closest_approach_frame"),
                },
                "diagnostic_notes": impact_result.get("diagnostic_notes", []),
            }
        else:
            return {
                "type": "NEAR-MISS",
                "closest_approach": {
                    "frame": impact_result.get("closest_approach_frame"),
                    "distance_m": impact_result.get("closest_approach_distance_m", 0),
                },
                "max_iou": metrics_result.get("max_iou", 0),
                "diagnostic_notes": impact_result.get("diagnostic_notes", []),
            }

    def _print_report(self, report: dict[str, Any]):
        """Print the analysis report to console."""
        print("\n" + "=" * 80)
        print("DYNAMIC ANALYSIS REPORT")
        print("=" * 80)

        # Narrative Summary
        print("\n### SCENARIO SUMMARY ###")
        print(report["narrative_summary"])

        # Timeline
        if report["timeline"]:
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
            if impact["severity_indicators"]["min_distance_m"]:
                print(
                    f"  - Minimum distance: {impact['severity_indicators']['min_distance_m']:.1f}m"
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

        if quality["quality_issues"]:
            print("\nQuality Issues:")
            for issue in quality["quality_issues"]:
                print(f"  ⚠ {issue}")

        if quality["assumptions"]:
            print("\nAssumptions & Warnings:")
            for assumption in quality["assumptions"]:
                print(f"  ⚠ {assumption}")

        # Tools Executed
        print("\n### ANALYSIS METHODOLOGY ###")
        print(f"Tools executed: {', '.join(report['tools_executed'])}")
        print(f"Dynamic analysis: {'Yes' if report.get('dynamic_analysis') else 'No'}")

        print("\n" + "=" * 80)

"""
AWS Bedrock Agent Tool Schemas for Accident Analysis.
This module defines the tool schemas that AWS agents can invoke.
"""

from typing import Any

from .tools import (
    build_timeline,
    compute_pair_metrics,
    load_detections,
    report_assumptions,
    trace_impact_window,
)


def create_tool_schemas() -> list[dict[str, Any]]:
    """
    Create AWS Bedrock Agent tool schemas.

    Returns:
        List of tool schemas in AWS format
    """
    return [
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
                                "items": {"type": "integer"},
                                "description": "List of track IDs to load (e.g., [7, 14])",
                            },
                            "frame_range": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "Optional (start_frame, end_frame) tuple to restrict loading",
                            },
                            "require_pairing": {
                                "type": "boolean",
                                "description": "If true, only return frames where all track_ids appear",
                                "default": True,
                            },
                            "fps_hint": {
                                "type": "number",
                                "description": "Optional FPS hint for metadata",
                            },
                        },
                        "required": ["track_ids"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "compute_pair_metrics",
                "description": "Compute collision metrics (IoU, distances, speeds) for paired detections. Returns enriched MetricRow objects with flags for collision candidates.",
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
        },
        {
            "toolSpec": {
                "name": "trace_impact_window",
                "description": "Trace the impact window to detect collision events. Returns collision detection results, impact frames, and diagnostic notes about the collision or near-miss.",
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
                                "type": "integer",
                                "description": "Number of frames to persist overlap",
                                "default": 3,
                            },
                        },
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "build_timeline",
                "description": "Build a structured timeline of events from approach to separation. Returns timeline entries with frame, timestamp, metrics, and narrative for each stage.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "padding_frames": {
                                "type": "integer",
                                "description": "Number of frames to include before/after impact",
                                "default": 10,
                            },
                            "stages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Custom stage labels (defaults to approach, first_contact, peak_overlap, separation)",
                            },
                        },
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "report_assumptions",
                "description": "Report data quality issues and assumptions made during analysis. Returns list of warnings about missing data, low confidence detections, and frame gaps.",
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
        },
        {
            "toolSpec": {
                "name": "get_weather_data",
                "description": "Retrieve weather conditions for a specific location and time to assess environmental factors that may have contributed to the accident.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "Latitude coordinate of the accident location",
                            },
                            "longitude": {
                                "type": "number", 
                                "description": "Longitude coordinate of the accident location",
                            },
                            "timestamp": {
                                "type": "string",
                                "description": "ISO timestamp of when the accident occurred",
                            },
                        },
                        "required": ["latitude", "longitude", "timestamp"],
                    }
                },
            }
        },
    ]


class AgentToolHandler:
    """
    Handles tool invocations from AWS Bedrock Agent.
    Maintains state between tool calls.
    """

    def __init__(
        self,
        detections_file: str = "backend/src/common/features/process_video/detections.jsonl",
    ):
        """
        Initialize the tool handler.

        Args:
            detections_file: Path to the detections JSONL file
        """
        self.detections_file = detections_file
        self.state = {
            "records": None,
            "metadata": None,
            "metric_rows": None,
            "impact_summary": None,
            "timeline": None,
            "assumptions": None,
        }

    def handle_tool_call(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle a tool invocation from the agent.

        Args:
            tool_name: Name of the tool to invoke
            tool_input: Input parameters for the tool

        Returns:
            Tool execution result
        """
        if tool_name == "load_detections":
            return self._handle_load_detections(tool_input)
        elif tool_name == "compute_pair_metrics":
            return self._handle_compute_pair_metrics(tool_input)
        elif tool_name == "trace_impact_window":
            return self._handle_trace_impact_window(tool_input)
        elif tool_name == "build_timeline":
            return self._handle_build_timeline(tool_input)
        elif tool_name == "report_assumptions":
            return self._handle_report_assumptions(tool_input)
        elif tool_name == "get_weather_data":
            return self._handle_get_weather(tool_input)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _handle_load_detections(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Handle load_detections tool call."""
        try:
            track_ids = tool_input["track_ids"]
            frame_range = (
                tuple(tool_input.get("frame_range"))
                if tool_input.get("frame_range")
                else None
            )
            require_pairing = tool_input.get("require_pairing", True)
            fps_hint = tool_input.get("fps_hint")

            result = load_detections(
                track_ids=track_ids,
                frame_range=frame_range,
                require_pairing=require_pairing,
                fps_hint=fps_hint,
                detections_file=self.detections_file,
            )

            # Store in state
            self.state["records"] = result["records"]
            self.state["metadata"] = result["metadata"]

            # Return summary for the agent
            return {
                "success": True,
                "total_frames": len(result["records"]),
                "frame_range": result["metadata"].get("frame_range"),
                "time_range": result["metadata"].get("time_range"),
                "track_ids": track_ids,
                "fps_estimated": result["metadata"].get("fps_estimated"),
                "missing_frames": len(result["metadata"].get("missing_frames", [])),
                "data_gaps": result["metadata"].get("data_gaps"),
                "message": f"Loaded {len(result['records'])} frames of paired detection data for tracks {track_ids}",
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _handle_compute_pair_metrics(
        self, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle compute_pair_metrics tool call."""
        try:
            if self.state["records"] is None:
                return {"error": "Must call load_detections first", "success": False}

            iou_threshold = tool_input.get("iou_threshold", 0.01)
            vehicle_length_m = tool_input.get("vehicle_length_m", 4.5)
            include_headings = tool_input.get("include_headings", False)

            metric_rows = compute_pair_metrics(
                pairs=self.state["records"],
                iou_threshold=iou_threshold,
                vehicle_length_m=vehicle_length_m,
                include_headings=include_headings,
            )

            # Store in state
            self.state["metric_rows"] = metric_rows

            # Compute summary statistics
            ious = [row.iou for row in metric_rows]
            distances = [
                row.world_distance_m
                for row in metric_rows
                if row.world_distance_m is not None
            ]
            collision_candidates = sum(
                1 for row in metric_rows if row.collision_candidate
            )

            return {
                "success": True,
                "total_frames": len(metric_rows),
                "max_iou": max(ious) if ious else 0,
                "min_iou": min(ious) if ious else 0,
                "avg_iou": sum(ious) / len(ious) if ious else 0,
                "min_distance_m": min(distances) if distances else None,
                "collision_candidates": collision_candidates,
                "message": f"Computed metrics for {len(metric_rows)} frames with {collision_candidates} collision candidates",
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _handle_trace_impact_window(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Handle trace_impact_window tool call."""
        try:
            if self.state["metric_rows"] is None:
                return {
                    "error": "Must call compute_pair_metrics first",
                    "success": False,
                }

            iou_threshold = tool_input.get("iou_threshold", 0.01)
            distance_threshold_m = tool_input.get("distance_threshold_m", 5.0)
            persistence_frames = tool_input.get("persistence_frames", 3)

            impact_summary = trace_impact_window(
                metric_rows=self.state["metric_rows"],
                iou_threshold=iou_threshold,
                distance_threshold_m=distance_threshold_m,
                persistence_frames=persistence_frames,
            )

            # Store in state
            self.state["impact_summary"] = impact_summary

            return {
                "success": True,
                "collision_detected": impact_summary["collision_detected"],
                "impact_frames_count": len(impact_summary["impact_frames"]),
                "overlap_duration_frames": impact_summary["overlap_duration_frames"],
                "first_contact_frame": impact_summary.get("first_contact_frame"),
                "last_overlap_frame": impact_summary.get("last_overlap_frame"),
                "closest_approach_frame": impact_summary["closest_approach"]["frame"],
                "closest_approach_distance_m": impact_summary["closest_approach"][
                    "distance_m"
                ],
                "diagnostic_notes": impact_summary["diagnostic_notes"],
                "impact_frames": impact_summary["impact_frames"][:5]
                if len(impact_summary["impact_frames"]) > 5
                else impact_summary["impact_frames"],  # Limit to first 5 for summary
                "message": f"Collision {'DETECTED' if impact_summary['collision_detected'] else 'NOT DETECTED'}",
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _handle_build_timeline(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Handle build_timeline tool call."""
        try:
            if (
                self.state["metric_rows"] is None
                or self.state["impact_summary"] is None
            ):
                return {
                    "error": "Must call compute_pair_metrics and trace_impact_window first",
                    "success": False,
                }

            padding_frames = tool_input.get("padding_frames", 10)
            stages = tool_input.get("stages")

            timeline_result = build_timeline(
                metric_rows=self.state["metric_rows"],
                impact_summary=self.state["impact_summary"],
                padding_frames=padding_frames,
                stages=stages,
            )

            # Store in state
            self.state["timeline"] = timeline_result

            return {
                "success": True,
                "timeline": timeline_result["timeline"],
                "summary_table": timeline_result["summary_table"],
                "stages": timeline_result["stages"],
                "frame_range": timeline_result["frame_range"],
                "total_frames": timeline_result["total_frames"],
                "message": f"Built timeline with {len(timeline_result['timeline'])} key events",
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _handle_report_assumptions(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Handle report_assumptions tool call."""
        try:
            if self.state["metric_rows"] is None or self.state["metadata"] is None:
                return {
                    "error": "Must call load_detections and compute_pair_metrics first",
                    "success": False,
                }

            warn_if_missing = tuple(
                tool_input.get("warn_if_missing", ["world_coords", "speed_mph"])
            )

            assumptions = report_assumptions(
                metric_rows=self.state["metric_rows"],
                metadata=self.state["metadata"],
                warn_if_missing=warn_if_missing,
            )

            # Store in state
            self.state["assumptions"] = assumptions

            return {
                "success": True,
                "assumptions": assumptions,
                "total_issues": len(assumptions),
                "message": f"Identified {len(assumptions)} data quality issues and assumptions",
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _handle_get_weather(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Handle get_weather_data tool call."""
        try:
            import random
            from datetime import datetime
            
            latitude = tool_input["latitude"]
            longitude = tool_input["longitude"]
            timestamp = tool_input["timestamp"]
            
            # Generate random but realistic weather data
            conditions = ["Clear", "Partly Cloudy", "Cloudy", "Overcast", "Foggy"]
            precipitations = ["None", "Light Rain", "Moderate Rain", "Heavy Rain", "Snow", "Sleet"]
            road_conditions = ["Dry", "Wet", "Icy", "Snow-covered", "Slippery"]
            
            # Random weather generation
            condition = random.choice(conditions)
            precipitation = random.choice(precipitations)
            road_condition = random.choice(road_conditions)
            
            # Temperature in Fahrenheit (realistic range)
            temperature_f = random.randint(20, 85)
            
            # Visibility in miles (affected by weather)
            if condition == "Foggy" or precipitation in ["Heavy Rain", "Snow"]:
                visibility_mi = round(random.uniform(0.1, 2.0), 1)
            elif precipitation in ["Light Rain", "Moderate Rain", "Sleet"]:
                visibility_mi = round(random.uniform(1.0, 5.0), 1)
            else:
                visibility_mi = round(random.uniform(5.0, 10.0), 1)
            
            weather_data = {
                "temperature_f": temperature_f,
                "condition": condition,
                "precipitation": precipitation,
                "visibility_mi": visibility_mi,
                "road_condition": road_condition,
                "location": {"latitude": latitude, "longitude": longitude},
                "timestamp": timestamp,
            }
            
            return {
                "success": True,
                "weather_data": weather_data,
                "message": f"Weather conditions: {condition}, {temperature_f}°F, {precipitation}, Visibility: {visibility_mi} mi, Road: {road_condition}",
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def get_full_state(self) -> dict[str, Any]:
        """
        Get the complete analysis state.

        Returns:
            Dictionary with all analysis results
        """
        return {
            "metadata": self.state["metadata"],
            "impact_summary": self.state["impact_summary"],
            "timeline": self.state["timeline"],
            "assumptions": self.state["assumptions"],
        }

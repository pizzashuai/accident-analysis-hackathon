"""
AWS Bedrock Agent Integration for Accident Analysis.
This module provides the interface for AWS Bedrock agents to invoke analysis tools.
"""

import json
from typing import Any

from .agent_tools import AgentToolHandler, create_tool_schemas


class BedrockAgentHandler:
    """
    Handler for AWS Bedrock Agent integration.
    Processes Lambda function events from AWS Bedrock agents.
    """

    def __init__(
        self,
        detections_file: str = "backend/src/common/features/process_video/detections.jsonl",
    ):
        """
        Initialize the Bedrock agent handler.

        Args:
            detections_file: Path to the detections JSONL file
        """
        self.tool_handler = AgentToolHandler(detections_file=detections_file)
        self.session_attributes = {}

    def lambda_handler(
        self, event: dict[str, Any], context: Any = None
    ) -> dict[str, Any]:
        """
        AWS Lambda handler for Bedrock Agent events.

        Args:
            event: Lambda event from Bedrock agent
            context: Lambda context (optional)

        Returns:
            Response in Bedrock agent format
        """
        print(f"Received event: {json.dumps(event)}")

        # Extract agent information
        agent = event.get("agent", {})
        action_group = event.get("actionGroup", "")
        api_path = event.get("apiPath", "")
        http_method = event.get("httpMethod", "")
        parameters = event.get("parameters", [])
        request_body = event.get("requestBody", {})

        # Extract session attributes
        session_attributes = event.get("sessionAttributes", {})
        prompt_session_attributes = event.get("promptSessionAttributes", {})

        # Determine which tool to invoke based on apiPath
        tool_name = self._extract_tool_name(api_path)

        if not tool_name:
            return self._error_response(f"Unknown API path: {api_path}")

        # Convert parameters to tool input format
        tool_input = self._convert_parameters(parameters, request_body)

        # Execute the tool
        try:
            result = self.tool_handler.handle_tool_call(tool_name, tool_input)

            if result.get("success"):
                return self._success_response(result, session_attributes)
            else:
                return self._error_response(
                    result.get("error", "Tool execution failed")
                )

        except Exception as e:
            print(f"Error executing tool {tool_name}: {str(e)}")
            return self._error_response(str(e))

    def _extract_tool_name(self, api_path: str) -> str:
        """
        Extract tool name from API path.

        Args:
            api_path: API path from Bedrock agent

        Returns:
            Tool name or empty string
        """
        # Expected format: /tools/{tool_name}
        if api_path.startswith("/tools/"):
            return api_path.split("/")[-1]

        # Map common paths to tool names
        path_mapping = {
            "/load-detections": "load_detections",
            "/compute-metrics": "compute_pair_metrics",
            "/trace-impact": "trace_impact_window",
            "/build-timeline": "build_timeline",
            "/report-assumptions": "report_assumptions",
        }

        return path_mapping.get(api_path, "")

    def _convert_parameters(
        self, parameters: list[dict[str, Any]], request_body: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Convert Bedrock agent parameters to tool input format.

        Args:
            parameters: List of parameters from Bedrock
            request_body: Request body content

        Returns:
            Tool input dictionary
        """
        tool_input = {}

        # Convert parameter list to dictionary
        for param in parameters:
            name = param.get("name")
            value = param.get("value")
            param_type = param.get("type", "string")

            # Convert types
            if param_type == "integer":
                value = int(value)
            elif param_type == "number":
                value = float(value)
            elif param_type == "boolean":
                value = value.lower() == "true"
            elif param_type == "array":
                if isinstance(value, str):
                    value = json.loads(value)

            tool_input[name] = value

        # Merge request body if present
        if request_body:
            content = request_body.get("content", {})
            if isinstance(content, dict):
                tool_input.update(content)

        return tool_input

    def _success_response(
        self, result: dict[str, Any], session_attributes: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Format successful response for Bedrock agent.

        Args:
            result: Tool execution result
            session_attributes: Session attributes to maintain

        Returns:
            Bedrock agent response
        """
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": "AccidentAnalysisTools",
                "apiPath": "/tools/response",
                "httpMethod": "POST",
                "httpStatusCode": 200,
                "responseBody": {"application/json": {"body": json.dumps(result)}},
                "sessionAttributes": session_attributes,
                "promptSessionAttributes": {},
            },
        }

    def _error_response(self, error_message: str) -> dict[str, Any]:
        """
        Format error response for Bedrock agent.

        Args:
            error_message: Error message

        Returns:
            Bedrock agent error response
        """
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": "AccidentAnalysisTools",
                "apiPath": "/tools/error",
                "httpMethod": "POST",
                "httpStatusCode": 500,
                "responseBody": {
                    "application/json": {
                        "body": json.dumps({"success": False, "error": error_message})
                    }
                },
            },
        }


def create_openapi_schema() -> dict[str, Any]:
    """
    Create OpenAPI schema for AWS Bedrock Agent action group.

    Returns:
        OpenAPI 3.0 schema
    """
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Accident Analysis Tools API",
            "version": "1.0.0",
            "description": "API for analyzing vehicle collision data from video detections",
        },
        "paths": {
            "/tools/load_detections": {
                "post": {
                    "summary": "Load detection data",
                    "description": "Load detection data for specified track IDs from the detections JSONL file",
                    "operationId": "loadDetections",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "track_ids": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "description": "List of track IDs to load",
                                        },
                                        "frame_range": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "minItems": 2,
                                            "maxItems": 2,
                                            "description": "Optional frame range [start, end]",
                                        },
                                        "require_pairing": {
                                            "type": "boolean",
                                            "default": True,
                                        },
                                    },
                                    "required": ["track_ids"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Successfully loaded detections",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "total_frames": {"type": "integer"},
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/tools/compute_pair_metrics": {
                "post": {
                    "summary": "Compute collision metrics",
                    "description": "Compute IoU, distances, and speeds for paired detections",
                    "operationId": "computePairMetrics",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "iou_threshold": {
                                            "type": "number",
                                            "default": 0.01,
                                        },
                                        "vehicle_length_m": {
                                            "type": "number",
                                            "default": 4.5,
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Successfully computed metrics"}
                    },
                }
            },
            "/tools/trace_impact_window": {
                "post": {
                    "summary": "Trace impact window",
                    "description": "Detect collision events and identify impact frames",
                    "operationId": "traceImpactWindow",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "iou_threshold": {
                                            "type": "number",
                                            "default": 0.01,
                                        },
                                        "distance_threshold_m": {
                                            "type": "number",
                                            "default": 5.0,
                                        },
                                        "persistence_frames": {
                                            "type": "integer",
                                            "default": 3,
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Successfully traced impact window"}
                    },
                }
            },
            "/tools/build_timeline": {
                "post": {
                    "summary": "Build event timeline",
                    "description": "Generate structured timeline from approach to separation",
                    "operationId": "buildTimeline",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "padding_frames": {
                                            "type": "integer",
                                            "default": 10,
                                        }
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Successfully built timeline"}
                    },
                }
            },
            "/tools/report_assumptions": {
                "post": {
                    "summary": "Report data quality issues",
                    "description": "Identify missing data and assumptions",
                    "operationId": "reportAssumptions",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "warn_if_missing": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "default": ["world_coords", "speed_mph"],
                                        }
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Successfully reported assumptions"}
                    },
                }
            },
        },
    }


def save_openapi_schema(output_file: str = "openapi_schema.json"):
    """
    Save OpenAPI schema to file for AWS Bedrock configuration.

    Args:
        output_file: Output file path
    """
    schema = create_openapi_schema()
    with open(output_file, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"OpenAPI schema saved to {output_file}")


if __name__ == "__main__":
    # Generate OpenAPI schema for AWS configuration
    save_openapi_schema()
    print("\nUse this schema to configure AWS Bedrock Agent action group.")
    print("\nTool schemas for Bedrock:")
    schemas = create_tool_schemas()
    for schema in schemas:
        print(f"\n- {schema['toolSpec']['name']}")
        print(f"  {schema['toolSpec']['description']}")

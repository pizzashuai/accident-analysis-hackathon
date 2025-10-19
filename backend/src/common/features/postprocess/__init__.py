"""
Postprocessing tools and agent for accident analysis.

This package provides:
- Tools for analyzing collision data (tools.py)
- AWS Bedrock agent integration (agent_aws.py, agent_tools.py)
- Local agent orchestration (agent_core.py)
- Main entry point for testing (main.py)
"""

from .tools import (
    load_detections,
    compute_pair_metrics,
    trace_impact_window,
    build_timeline,
    report_assumptions,
    Detection,
    MetricRow
)

from .agent_core import AccidentAnalysisAgent, AgentConfig
from .agent_tools import AgentToolHandler, create_tool_schemas
from .agent_aws import BedrockAgentHandler, create_openapi_schema

__all__ = [
    # Core tools
    'load_detections',
    'compute_pair_metrics',
    'trace_impact_window',
    'build_timeline',
    'report_assumptions',
    'Detection',
    'MetricRow',
    # Agent components
    'AccidentAnalysisAgent',
    'AgentConfig',
    'AgentToolHandler',
    'BedrockAgentHandler',
    'create_tool_schemas',
    'create_openapi_schema',
]

#!/usr/bin/env python3
"""
Test script for LLM-powered accident analysis agent.
This script demonstrates the agent functionality without requiring AWS credentials.
"""

from llm_agent import LLMAccidentAnalysisAgent, LLMAgentConfig


def test_agent_without_aws():
    """Test the agent initialization and tool functionality without AWS."""
    print("=" * 80)
    print("TESTING LLM AGENT FUNCTIONALITY")
    print("=" * 80)

    # Create test configuration
    config = LLMAgentConfig(
        track_ids=[7, 14],
        frame_range=None,
        iou_threshold=0.01,
        distance_threshold_m=5.0,
        persistence_frames=3,
        padding_frames=10,
        detections_file="../process_video/detections.jsonl",
        aws_region="us-east-1",
        bedrock_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_iterations=20,
        temperature=0.0,
        max_tokens=4096,
    )

    print("Configuration created:")
    print(f"  Track IDs: {config.track_ids}")
    print(f"  Detections file: {config.detections_file}")
    print(f"  AWS Region: {config.aws_region}")
    print(f"  Bedrock Model: {config.bedrock_model_id}")

    # Test agent initialization
    try:
        agent = LLMAccidentAnalysisAgent(config)
        print("\n✓ Agent initialized successfully")

        # Test tool handler initialization
        if agent.tool_handler:
            print("✓ Tool handler initialized successfully")

            # Test loading detections (without AWS)
            print("\nTesting tool functionality...")
            result = agent.tool_handler.handle_tool_call(
                "load_detections",
                {
                    "track_ids": [7, 14],
                    "frame_range": None,
                    "require_pairing": True,
                },
            )

            if result.get("success"):
                print("✓ load_detections tool works")
                print(f"  Loaded {result.get('total_frames', 0)} frames")
                print(f"  Frame range: {result.get('frame_range')}")
                if result.get("time_range"):
                    print(
                        f"  Time span: {result['time_range'][0]:.3f}s - {result['time_range'][1]:.3f}s"
                    )
            else:
                print(
                    f"✗ load_detections failed: {result.get('error', 'Unknown error')}"
                )

        else:
            print("✗ Tool handler not initialized")

    except Exception as e:
        print(f"✗ Agent initialization failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    # Test AWS client initialization (should fail gracefully)
    print(
        f"\nAWS Bedrock client status: {'Available' if agent.bedrock_client else 'Not available (expected without credentials)'}"
    )

    if not agent.bedrock_client:
        print("This is expected behavior when AWS credentials are not configured.")
        print("To use the full LLM functionality, configure AWS credentials with:")
        print("  aws configure")
        print(
            "  or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        )

    return True


def test_tool_functionality():
    """Test individual tool functions."""
    print("\n" + "=" * 80)
    print("TESTING INDIVIDUAL TOOLS")
    print("=" * 80)

    from agent_tools import AgentToolHandler

    # Initialize tool handler
    tool_handler = AgentToolHandler(detections_file="../process_video/detections.jsonl")

    # Test each tool
    tools_to_test = [
        (
            "load_detections",
            {
                "track_ids": [7, 14],
                "frame_range": None,
                "require_pairing": True,
            },
        ),
        (
            "compute_pair_metrics",
            {
                "iou_threshold": 0.01,
                "vehicle_length_m": 4.5,
                "include_headings": False,
            },
        ),
        (
            "trace_impact_window",
            {
                "iou_threshold": 0.01,
                "distance_threshold_m": 5.0,
                "persistence_frames": 3,
            },
        ),
        (
            "build_timeline",
            {
                "padding_frames": 10,
            },
        ),
        (
            "report_assumptions",
            {
                "warn_if_missing": ["world_coords", "speed_mph"],
            },
        ),
    ]

    for tool_name, tool_input in tools_to_test:
        print(f"\nTesting {tool_name}...")
        try:
            result = tool_handler.handle_tool_call(tool_name, tool_input)
            if result.get("success"):
                print(f"✓ {tool_name} completed successfully")
                # Show some key results
                if tool_name == "load_detections":
                    print(f"  Frames loaded: {result.get('total_frames', 0)}")
                elif tool_name == "compute_pair_metrics":
                    print(f"  Max IoU: {result.get('max_iou', 0):.3f}")
                    print(f"  Min distance: {result.get('min_distance_m', 'N/A')}")
                elif tool_name == "trace_impact_window":
                    collision = result.get("collision_detected", False)
                    print(f"  Collision detected: {collision}")
                elif tool_name == "build_timeline":
                    timeline = result.get("timeline", [])
                    print(f"  Timeline events: {len(timeline)}")
                elif tool_name == "report_assumptions":
                    assumptions = result.get("assumptions", [])
                    print(f"  Assumptions: {len(assumptions)}")
            else:
                print(f"✗ {tool_name} failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"✗ {tool_name} error: {str(e)}")


def main():
    """Run all tests."""
    print("LLM-Powered Accident Analysis Agent - Test Suite")
    print("=" * 80)

    # Test agent initialization
    if not test_agent_without_aws():
        print("\n❌ Agent test failed")
        return 1

    # Test tool functionality
    test_tool_functionality()

    print("\n" + "=" * 80)
    print("✓ ALL TESTS COMPLETED")
    print("=" * 80)
    print(
        "\nThe agent is ready to use with AWS Bedrock when credentials are configured."
    )
    print("Run 'python main.py --track-ids 7 14' to analyze accident data.")

    return 0


if __name__ == "__main__":
    import sys

    exit_code = main()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
Test script to demonstrate dynamic vs static agent analysis.
This script compares the two approaches side by side.
"""

from pathlib import Path

from agent_core import AccidentAnalysisAgent, AgentConfig


def test_dynamic_vs_static():
    """Test both dynamic and static analysis approaches."""
    print("=" * 80)
    print("DYNAMIC vs STATIC AGENT COMPARISON")
    print("=" * 80)

    # Test configuration
    track_ids = [7, 14]
    detections_file = "../process_video/detections.jsonl"

    # Check if detections file exists
    detections_path = Path(detections_file)
    if not detections_path.exists():
        script_dir = Path(__file__).parent
        detections_path = script_dir / detections_file
        if not detections_path.exists():
            print(f"❌ Detections file not found: {detections_file}")
            return

    print(f"\nTesting with tracks: {track_ids}")
    print(f"Detections file: {detections_path}")

    # Test 1: Static Analysis
    print("\n" + "=" * 60)
    print("TEST 1: STATIC ANALYSIS (Hardcoded Workflow)")
    print("=" * 60)

    static_config = AgentConfig(
        track_ids=track_ids,
        frame_range=None,
        iou_threshold=0.01,
        distance_threshold_m=5.0,
        persistence_frames=3,
        padding_frames=10,
        detections_file=str(detections_path),
        use_dynamic_analysis=False,
        include_speed_analysis=True,
        include_heading_analysis=False,
        custom_timeline_stages=None,
        fps_hint=None,
        vehicle_length_m=4.5,
    )

    try:
        static_agent = AccidentAnalysisAgent(static_config)
        static_report = static_agent.analyze()

        if static_report.get("success"):
            print("\n✓ Static analysis completed successfully")
            print(f"  Collision detected: {static_report['collision_detected']}")
            print(f"  Tools executed: {static_report.get('tools_executed', 'N/A')}")
            print(f"  Timeline events: {len(static_report.get('timeline', []))}")
        else:
            print(
                f"\n❌ Static analysis failed: {static_report.get('error', 'Unknown error')}"
            )

    except Exception as e:
        print(f"\n❌ Static analysis error: {str(e)}")

    # Test 2: Dynamic Analysis
    print("\n" + "=" * 60)
    print("TEST 2: DYNAMIC ANALYSIS (Intelligent Tool Selection)")
    print("=" * 60)

    dynamic_config = AgentConfig(
        track_ids=track_ids,
        frame_range=None,
        iou_threshold=0.01,
        distance_threshold_m=5.0,
        persistence_frames=3,
        padding_frames=10,
        detections_file=str(detections_path),
        use_dynamic_analysis=True,
        include_speed_analysis=True,
        include_heading_analysis=False,
        custom_timeline_stages=None,
        fps_hint=None,
        vehicle_length_m=4.5,
    )

    try:
        dynamic_agent = AccidentAnalysisAgent(dynamic_config)
        dynamic_report = dynamic_agent.analyze()

        if dynamic_report.get("success"):
            print("\n✓ Dynamic analysis completed successfully")
            print(f"  Collision detected: {dynamic_report['collision_detected']}")
            print(
                f"  Tools executed: {', '.join(dynamic_report.get('tools_executed', []))}"
            )
            print(f"  Timeline events: {len(dynamic_report.get('timeline', []))}")
            print(
                f"  Dynamic analysis: {dynamic_report.get('dynamic_analysis', False)}"
            )
        else:
            print(
                f"\n❌ Dynamic analysis failed: {dynamic_report.get('error', 'Unknown error')}"
            )

    except Exception as e:
        print(f"\n❌ Dynamic analysis error: {str(e)}")

    # Test 3: Dynamic Analysis with Heading Analysis
    print("\n" + "=" * 60)
    print("TEST 3: DYNAMIC ANALYSIS WITH HEADING ANALYSIS")
    print("=" * 60)

    heading_config = AgentConfig(
        track_ids=track_ids,
        frame_range=None,
        iou_threshold=0.01,
        distance_threshold_m=5.0,
        persistence_frames=3,
        padding_frames=10,
        detections_file=str(detections_path),
        use_dynamic_analysis=True,
        include_speed_analysis=True,
        include_heading_analysis=True,  # Enable heading analysis
        custom_timeline_stages=None,
        fps_hint=30.0,  # Provide FPS hint
        vehicle_length_m=4.5,
    )

    try:
        heading_agent = AccidentAnalysisAgent(heading_config)
        heading_report = heading_agent.analyze()

        if heading_report.get("success"):
            print("\n✓ Dynamic analysis with heading completed successfully")
            print(f"  Collision detected: {heading_report['collision_detected']}")
            print(
                f"  Tools executed: {', '.join(heading_report.get('tools_executed', []))}"
            )
            print(f"  Timeline events: {len(heading_report.get('timeline', []))}")
            print(f"  FPS hint used: {heading_config.fps_hint}")
        else:
            print(
                f"\n❌ Dynamic analysis with heading failed: {heading_report.get('error', 'Unknown error')}"
            )

    except Exception as e:
        print(f"\n❌ Dynamic analysis with heading error: {str(e)}")

    # Test 4: Dynamic Analysis with Custom Parameters
    print("\n" + "=" * 60)
    print("TEST 4: DYNAMIC ANALYSIS WITH STRICT THRESHOLDS")
    print("=" * 60)

    strict_config = AgentConfig(
        track_ids=track_ids,
        frame_range=(2, 20),  # Limited frame range
        iou_threshold=0.05,  # Stricter IoU threshold
        distance_threshold_m=2.0,  # Stricter distance threshold
        persistence_frames=5,  # More persistence required
        padding_frames=5,
        detections_file=str(detections_path),
        use_dynamic_analysis=True,
        include_speed_analysis=True,
        include_heading_analysis=False,
        custom_timeline_stages=["approach", "contact", "separation"],  # Custom stages
        fps_hint=None,
        vehicle_length_m=5.0,  # Longer vehicle assumption
    )

    try:
        strict_agent = AccidentAnalysisAgent(strict_config)
        strict_report = strict_agent.analyze()

        if strict_report.get("success"):
            print("\n✓ Dynamic analysis with strict thresholds completed successfully")
            print(f"  Collision detected: {strict_report['collision_detected']}")
            print(
                f"  Tools executed: {', '.join(strict_report.get('tools_executed', []))}"
            )
            print(f"  Timeline events: {len(strict_report.get('timeline', []))}")
            print(f"  Frame range: {strict_config.frame_range}")
            print(f"  Custom stages: {strict_config.custom_timeline_stages}")
        else:
            print(
                f"\n❌ Dynamic analysis with strict thresholds failed: {strict_report.get('error', 'Unknown error')}"
            )

    except Exception as e:
        print(f"\n❌ Dynamic analysis with strict thresholds error: {str(e)}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nKey Differences:")
    print("• Static Analysis: Uses hardcoded workflow with fixed tool sequence")
    print(
        "• Dynamic Analysis: Intelligently selects tools based on data and requirements"
    )
    print("• Dynamic Analysis: Can adapt parameters based on data quality")
    print("• Dynamic Analysis: Can skip unnecessary tools or add additional analysis")
    print("• Dynamic Analysis: Provides more detailed execution logging")


if __name__ == "__main__":
    test_dynamic_vs_static()

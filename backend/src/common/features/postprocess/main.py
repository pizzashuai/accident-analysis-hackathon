#!/usr/bin/env python3
"""
Main entry point for LLM-powered accident analysis.
This script uses AWS Bedrock Claude to analyze accident data with flexible output formats.
"""

import argparse
import json
from pathlib import Path

from llm_agent import LLMAccidentAnalysisAgent, LLMAgentConfig


def save_report(report: dict, output_file: str):
    """Save the analysis report to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to: {output_file}")


def main():
    """Main entry point for LLM-powered accident analysis."""
    parser = argparse.ArgumentParser(
        description="LLM-Powered Accident Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze collision between tracks 7 and 14
  python main.py --track-ids 7 14
  
  # Analyze specific frame range
  python main.py --track-ids 7 14 --frame-range 2 30
  
  # Use different AWS region and model
  python main.py --track-ids 7 14 --aws-region us-west-2 --bedrock-model anthropic.claude-3-5-sonnet-20241022-v2:0
  
  # Adjust detection thresholds
  python main.py --track-ids 7 14 --iou-threshold 0.02 --distance-threshold 3.0
  
  # Save report to file
  python main.py --track-ids 7 14 --output report.json
  
  # Use custom detections file
  python main.py --track-ids 7 14 --detections-file /path/to/detections.jsonl
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
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
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

    # Create LLM agent configuration
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

    print("\n" + "=" * 80)
    print("LLM-POWERED ACCIDENT ANALYSIS")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Track IDs: {config.track_ids}")
    print(
        f"  Frame range: {config.frame_range if config.frame_range else 'All frames'}"
    )
    print(f"  IoU threshold: {config.iou_threshold}")
    print(f"  Distance threshold: {config.distance_threshold_m}m")
    print(f"  Persistence frames: {config.persistence_frames}")
    print(f"  Padding frames: {config.padding_frames}")
    print(f"  Detections file: {detections_path}")
    print(f"  AWS Region: {config.aws_region}")
    print(f"  Bedrock Model: {config.bedrock_model_id}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max tokens: {config.max_tokens}")
    print()

    # Create and run LLM agent
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
            save_report(report, args.output)

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

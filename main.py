#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


from src.validate_video import is_video_valid


def main():
    parser = argparse.ArgumentParser(
        description="Parse a local video and print basic info as JSON."
    )
    parser.add_argument(
        "video", type=str, help="Path to video file (e.g., data/accident.mp4)"
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(json.dumps({"error": f"File not found: {video_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        info = is_video_valid(video_path)
        if args.pretty:
            print(json.dumps(info, indent=2))
        else:
            print(json.dumps(info, separators=(",", ":")))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

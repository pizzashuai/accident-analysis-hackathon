#!/usr/bin/env python3
"""
Test script for speed calculation with different smoothing techniques.
"""

import sys
import os
import logging

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from pathlib import Path

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Set paths
    jsonl_path = script_dir / "detections.jsonl"
    homography_path = script_dir / "homography-points.json"
    output_dir = script_dir / "speed_test_output"
    
    # Check if files exist
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found")
        sys.exit(1)
    
    if not homography_path.exists():
        print(f"Error: {homography_path} not found")
        sys.exit(1)
    
    print(f"Testing speed calculation with:")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Homography: {homography_path}")
    print(f"  Output: {output_dir}")
    print("")
    
    # Import processor after path is set
    from src.common.features.process_video.processor import main
    
    # Override sys.argv to pass arguments
    sys.argv = [
        "test_speed_smoothing.py",
        "--jsonl", str(jsonl_path),
        "--homography", str(homography_path),
        "--output-dir", str(output_dir),
        "--video-width", "1280",
        "--video-height", "720",
        "--max-speed", "100.0",  # Maximum reasonable speed (mph)
        "--min-speed", "0.0",
        "--lookback", "5",
    ]
    
    # Run main
    main()


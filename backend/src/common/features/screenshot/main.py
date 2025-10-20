"""Main testable script for screenshot and map overlay generation."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

from .video_screenshot import extract_screenshot, get_video_info
from .map_overlay import (
    generate_map_overlay, 
    calculate_map_center_and_zoom,
    filter_detections_by_timestamp,
    extract_trajectories_from_detections
)
from .collision_detector import (
    detect_collision_from_jsonl,
    get_collision_point_from_detections,
    analyze_detection_data_quality
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function for testing screenshot and map overlay generation."""
    parser = argparse.ArgumentParser(description="Generate video screenshots and map overlays for collision analysis")
    parser.add_argument(
        "--detections", 
        type=str,
        default="src/common/features/postprocess/detections-time.jsonl",
        help="Path to detections JSONL file"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="src/common/features/process_video/happy1.mp4",
        help="Path to video file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./screenshots",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Google Maps API key (if not provided, will try to get from config)"
    )
    parser.add_argument(
        "--track-ids",
        type=int,
        nargs="+",
        help="Specific track IDs to analyze (if not provided, will auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    detections_path = Path(args.detections)
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COLLISION ANALYSIS - SCREENSHOT & MAP OVERLAY GENERATION")
    print("=" * 80)
    
    # Step 1: Validate inputs
    print("\n[STEP 1] Validating inputs...")
    
    if not detections_path.exists():
        print(f"❌ Detections file not found: {detections_path}")
        return 1
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    print(f"✓ Detections file: {detections_path}")
    print(f"✓ Video file: {video_path}")
    print(f"✓ Output directory: {output_dir}")
    
    # Step 2: Get video info
    print("\n[STEP 2] Analyzing video...")
    video_info = get_video_info(video_path)
    
    if video_info.get("error"):
        print(f"❌ Error getting video info: {video_info['error']}")
        return 1
    
    print(f"✓ Video duration: {video_info['duration']:.2f}s")
    print(f"✓ Video FPS: {video_info['fps']:.2f}")
    print(f"✓ Total frames: {video_info['total_frames']}")
    print(f"✓ Resolution: {video_info['width']}x{video_info['height']}")
    
    # Step 3: Load and analyze detection data
    print("\n[STEP 3] Loading detection data...")
    
    with open(detections_path, 'r') as f:
        detections = []
        for line in f:
            if line.strip():
                detections.append(json.loads(line.strip()))
    
    print(f"✓ Loaded {len(detections)} detection records")
    
    # Analyze data quality
    quality_analysis = analyze_detection_data_quality(detections)
    print(f"✓ Data quality score: {quality_analysis['data_quality_score']:.1f}/100")
    print(f"✓ Track IDs found: {quality_analysis['unique_track_ids']}")
    
    # Step 4: Detect collision
    print("\n[STEP 4] Detecting collision...")
    
    collision_result = detect_collision_from_jsonl(
        jsonl_path=detections_path,
        track_ids=args.track_ids
    )
    
    if not collision_result["success"]:
        print(f"❌ Collision detection failed: {collision_result['error']}")
        return 1
    
    if not collision_result["collision_detected"]:
        print("⚠️  No collision detected in the data")
        print("Proceeding with closest approach analysis...")
        
        # Use closest approach frame
        impact_summary = collision_result["impact_summary"]
        closest_approach = impact_summary.get("closest_approach", {})
        collision_frame = closest_approach.get("frame")
        collision_timestamp = None
        
        # Find timestamp for closest approach frame
        for detection in detections:
            if detection.get('frame') == collision_frame:
                collision_timestamp = detection.get('time')
                break
        
        if collision_timestamp is None:
            print("❌ Could not find timestamp for closest approach frame")
            return 1
            
        print(f"✓ Closest approach at frame {collision_frame}, timestamp {collision_timestamp:.3f}s")
    else:
        collision_timestamp = collision_result["collision_timestamp"]
        collision_frame = collision_result["collision_frame"]
        print(f"✓ Collision detected at frame {collision_frame}, timestamp {collision_timestamp:.3f}s")
    
    # Step 5: Extract video screenshot
    print("\n[STEP 5] Extracting video screenshot...")
    
    screenshot_path = output_dir / f"collision_screenshot_frame_{collision_frame}.png"
    
    screenshot_result = extract_screenshot(
        video_path=video_path,
        timestamp=collision_timestamp,
        output_path=screenshot_path
    )
    
    if not screenshot_result["success"]:
        print(f"❌ Screenshot extraction failed: {screenshot_result['error']}")
        return 1
    
    print(f"✓ Screenshot saved: {screenshot_result['output_path']}")
    print(f"✓ Frame number: {screenshot_result['frame_number']}")
    
    # Step 6: Generate map overlay
    print("\n[STEP 6] Generating map overlay...")
    
    # Get Google Maps API key
    api_key = args.api_key
    if not api_key:
        try:
            from ...config import settings
            api_key = settings.GOOGLE_MAP_API_KEY
            print("✓ Using Google Maps API key from settings")
        except Exception as e:
            print(f"❌ Could not get API key from settings: {e}")
            print("Please provide --api-key argument")
            return 1
    
    # Filter detections for collision timestamp
    collision_detections = filter_detections_by_timestamp(detections, collision_timestamp)
    
    if not collision_detections:
        print("❌ No detections found at collision timestamp")
        return 1
    
    # Extract trajectories for the track IDs involved in collision
    track_ids = collision_result["track_ids"]
    trajectories = extract_trajectories_from_detections(detections, track_ids)
    
    if not trajectories:
        print("❌ No trajectories extracted")
        return 1
    
    print(f"✓ Extracted {len(trajectories)} vehicle trajectories")
    
    # Get collision point
    collision_point = get_collision_point_from_detections(detections, collision_frame)
    
    if collision_point[0] is None:
        print("❌ Could not determine collision point coordinates")
        return 1
    
    print(f"✓ Collision point: {collision_point[0]:.6f}, {collision_point[1]:.6f}")
    
    # Calculate map center and zoom
    center_lat, center_lng, zoom = calculate_map_center_and_zoom(trajectories, collision_point)
    
    print(f"✓ Map center: {center_lat:.6f}, {center_lng:.6f}")
    print(f"✓ Map zoom: {zoom}")
    
    # Generate map overlay
    map_path = output_dir / f"collision_map_frame_{collision_frame}.png"
    
    map_result = generate_map_overlay(
        center_lat=center_lat,
        center_lng=center_lng,
        trajectories=trajectories,
        collision_point=collision_point,
        api_key=api_key,
        output_path=map_path
    )
    
    if not map_result["success"]:
        print(f"❌ Map overlay generation failed: {map_result['error']}")
        return 1
    
    print(f"✓ Map overlay saved: {map_result['output_path']}")
    
    # Step 7: Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    print(f"\n📸 Screenshot: {screenshot_result['output_path']}")
    print(f"🗺️  Map overlay: {map_result['output_path']}")
    
    if collision_result["collision_detected"]:
        print(f"\n💥 Collision detected at:")
        print(f"   Frame: {collision_frame}")
        print(f"   Timestamp: {collision_timestamp:.3f}s")
        print(f"   Track IDs: {track_ids}")
    else:
        print(f"\n⚠️  No collision detected, closest approach at:")
        print(f"   Frame: {collision_frame}")
        print(f"   Timestamp: {collision_timestamp:.3f}s")
        print(f"   Track IDs: {track_ids}")
    
    print(f"\n📊 Data Quality: {quality_analysis['data_quality_score']:.1f}/100")
    print(f"📈 Total detections: {quality_analysis['total_detections']}")
    
    return 0


if __name__ == "__main__":
    exit(main())

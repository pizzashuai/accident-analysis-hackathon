#!/usr/bin/env python3
"""
Accident Analysis using Roboflow Supervision
Replaces custom detection, tracking, and annotation with supervision library.
"""

import argparse
import cv2
import json
import sys
from pathlib import Path
from typing import Optional
import re

import supervision as sv
from ultralytics import YOLO


def get_next_test_folder_number(out_dir: Path) -> int:
    """
    Find the next test folder number by looking for existing test_N folders.

    Args:
        out_dir: Path to the output directory

    Returns:
        Next test folder number (starting from 1 if no test folders exist)
    """
    if not out_dir.exists():
        return 1

    # Find all test_N folders
    test_folders = []
    for item in out_dir.iterdir():
        if item.is_dir() and item.name.startswith("test_"):
            # Extract number from test_N format
            match = re.match(r"test_(\d+)", item.name)
            if match:
                test_folders.append(int(match.group(1)))

    if not test_folders:
        return 1

    return max(test_folders) + 1


def validate_video(video_path: Path) -> dict:
    """Validate video file and return basic info."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    cap.release()

    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError("Video probe failed (width/height/fps/frames invalid).")

    return {
        "video": str(video_path.resolve()),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_count,
        "ok": True,
    }


def extract_video_info(video_path: Path) -> dict:
    """
    Extract video information including properties needed for processing.

    Args:
        video_path: Path to input video

    Returns:
        Dictionary containing video properties
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }


def process_video_with_supervision(
    video_path: Path,
    test_folder: Path,
    model_path: str = "yolov8s.pt",
    conf_threshold: float = 0.2,
    iou_threshold: float = 0.5,
    classes: Optional[list] = None,
    trail_length: int = 10,
    # Tracking parameters
    minimum_consecutive_frames: int = 2,
    track_activation_threshold: float = 0.1,
    lost_track_buffer: int = 100,
    minimum_matching_threshold: float = 0.95,
):
    """
    Process video using supervision for detection, tracking, and annotation.

    Args:
        video_path: Path to input video
        test_folder: Path to test folder where output files will be saved
        model_path: Path to YOLO model
        conf_threshold: Detection confidence threshold
        iou_threshold: IoU threshold for NMS
        classes: List of class IDs to detect (None for all)
        trail_length: Length of tracking trails
        minimum_consecutive_frames: Minimum frames for track activation
        track_activation_threshold: Threshold for track activation
        lost_track_buffer: Buffer for lost tracks
        minimum_matching_threshold: Minimum threshold for track matching
    """

    # Default vehicle classes (COCO): car, motorcycle, bus, truck
    if classes is None:
        classes = [2, 3, 5, 7]

    # Create test folder
    test_folder.mkdir(parents=True, exist_ok=True)

    # Set up output paths
    output_path = test_folder / "overlay_supervision.mp4"
    info_path = test_folder / "video_info.json"

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Opening video: {video_path}")
    # Extract video information
    video_info = extract_video_info(video_path)
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    total_frames = video_info["total_frames"]

    # Open video for processing
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize supervision components
    byte_tracker = sv.ByteTrack(
        minimum_consecutive_frames=minimum_consecutive_frames,
        frame_rate=fps,
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
    )
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Store tracking history for trails
    tracking_history = {}

    frame_count = 0
    processed_frames = 0

    print("Processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(
            frame,
            conf=conf_threshold,
            classes=classes,
            verbose=False,
            iou=iou_threshold,
        )[0]

        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)

        # Update tracker
        detections = byte_tracker.update_with_detections(detections)

        # Update tracking history for trails
        if detections.tracker_id is not None:
            for detection_idx in range(len(detections)):
                tracker_id = detections.tracker_id[detection_idx]
                if tracker_id is not None:
                    if tracker_id not in tracking_history:
                        tracking_history[tracker_id] = []

                    # Get center point of bounding box
                    x1, y1, x2, y2 = detections.xyxy[detection_idx]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    tracking_history[tracker_id].append((center_x, center_y))

                    # Keep only the last trail_length points
                    if len(tracking_history[tracker_id]) > trail_length:
                        tracking_history[tracker_id] = tracking_history[tracker_id][
                            -trail_length:
                        ]

        # Create labels with tracker IDs
        labels = []
        if detections.tracker_id is not None:
            for detection_idx in range(len(detections)):
                tracker_id = detections.tracker_id[detection_idx]
                if tracker_id is not None:
                    labels.append(f"ID:{tracker_id}")
                else:
                    labels.append("")
        else:
            labels = [""] * len(detections)

        # Annotate frame
        annotated_frame = frame.copy()

        # Draw bounding boxes and labels
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # Draw trails manually using OpenCV
        for tracker_id, trace_points in tracking_history.items():
            if len(trace_points) > 1:
                # Get color for this tracker ID
                color = sv.ColorPalette.DEFAULT.by_idx(tracker_id % 20)
                color_bgr = (int(color.b), int(color.g), int(color.r))

                # Draw lines connecting the trail points
                for i in range(1, len(trace_points)):
                    pt1 = trace_points[i - 1]
                    pt2 = trace_points[i]
                    cv2.line(annotated_frame, pt1, pt2, color_bgr, 2)

        # Write frame
        out.write(annotated_frame)
        processed_frames += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

        frame_count += 1

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Processing complete!")
    print(f"Output video saved to: {output_path}")
    print(f"Processed {processed_frames} frames")

    # Save video info
    video_info = {
        "video": str(video_path.resolve()),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": total_frames,
        "processed_frames": processed_frames,
        "model": model_path,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "classes": classes,
        "trail_length": trail_length,
        "tracking_params": {
            "minimum_consecutive_frames": minimum_consecutive_frames,
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
        },
    }

    with info_path.open("w") as f:
        json.dump(video_info, f, indent=2)

    print(f"Video info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Accident Analysis using Roboflow Supervision"
    )
    parser.add_argument("video", type=str, help="Path to video file (e.g., happy1.mp4)")
    parser.add_argument(
        "--output",
        type=str,
        default="out/overlay_supervision.mp4",
        help="Output video path (default: out/overlay_supervision.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="YOLO model path (default: yolov8s.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU threshold for NMS (default: 0.5)"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[2, 3, 5, 7],
        help="Class IDs to detect (default: 2,3,5,7 for vehicles)",
    )
    parser.add_argument(
        "--trail", type=int, default=10, help="Trail length in frames (default: 10)"
    )
    parser.add_argument(
        "--info-only", action="store_true", help="Only show video info, don't process"
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(json.dumps({"error": f"File not found: {video_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        # Validate video
        info = validate_video(video_path)

        if args.info_only:
            if args.pretty:
                print(json.dumps(info, indent=2))
            else:
                print(json.dumps(info, separators=(",", ":")))
            return

        # Create test folder
        out_dir = Path("out")
        test_number = get_next_test_folder_number(out_dir)
        test_folder = out_dir / f"test_{test_number}"

        print(f"Creating test folder: {test_folder}")

        process_video_with_supervision(
            video_path=video_path,
            test_folder=test_folder,
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            classes=args.classes,
            trail_length=args.trail,
        )

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

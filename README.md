# Accident Analysis with Roboflow Supervision

This project has been refactored to use [Roboflow Supervision](https://github.com/roboflow/supervision) for detection, tracking, and annotation, replacing the custom implementations with a more robust and maintainable solution.

## Features

- **Object Detection**: Uses YOLOv8 for vehicle detection (cars, motorcycles, buses, trucks)
- **Multi-Object Tracking**: Uses ByteTrack algorithm from supervision for consistent ID tracking
- **Visual Annotation**: Draws bounding boxes, labels, and tracking trails
- **Video Processing**: Processes entire videos with overlayed tracking information

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Usage

### Basic Usage

Process a video with default settings:

```bash
uv run python main.py happy1.mp4
```

This will create `out/overlay_supervision.mp4` with the processed video.

### Custom Output

Specify a custom output path:

```bash
uv run python main.py happy1.mp4 --output my_output.mp4
```

### Video Information Only

Get video information without processing:

```bash
uv run python main.py happy1.mp4 --info-only --pretty
```

### Advanced Options

```bash
uv run python main.py happy1.mp4 \
  --output out/custom_output.mp4 \
  --model yolov8s.pt \
  --conf 0.5 \
  --iou 0.4 \
  --classes 2 3 5 7 \
  --trail 15
```

### Parameters

- `--output`: Output video path (default: `out/overlay_supervision.mp4`)
- `--model`: YOLO model path (default: `yolov8n.pt`)
- `--conf`: Detection confidence threshold (default: 0.3)
- `--iou`: IoU threshold for NMS (default: 0.5)
- `--classes`: Class IDs to detect (default: 2,3,5,7 for vehicles)
- `--trail`: Trail length in frames (default: 10)
- `--info-only`: Only show video info, don't process
- `--pretty`: Pretty-print JSON output

## Output

The script generates:

1. **Overlayed Video**: Video with bounding boxes, object IDs, and tracking trails
2. **Video Info**: JSON file with processing metadata (`out/video_info.json`)

## What Was Replaced

The following custom implementations were replaced with supervision:

- ✅ **Detection**: Custom YOLO wrapper → `sv.Detections.from_ultralytics()`
- ✅ **Tracking**: Custom ByteTracker → `sv.ByteTrack()`
- ✅ **Annotation**: Custom overlay code → `sv.BoxAnnotator()` and `sv.LabelAnnotator()`
- ✅ **Trails**: Custom trail drawing → OpenCV-based trail rendering

## Benefits

- **Simplified Code**: Reduced from ~800 lines to ~250 lines
- **Better Tracking**: More robust ByteTrack implementation
- **Maintainable**: Uses well-maintained supervision library
- **Extensible**: Easy to add new features using supervision's ecosystem
- **Same Output**: Produces the same overlayed video format

## Dependencies

- `ultralytics>=8.2.0`: YOLO model inference
- `supervision>=0.20.0`: Detection, tracking, and annotation
- `opencv-python>=4.10.0`: Video processing
- `numpy>=1.26`: Numerical operations

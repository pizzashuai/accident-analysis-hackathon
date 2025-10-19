# Postprocessing Tools for Accident Analysis

This directory contains tools designed to be used by LLM agents for analyzing vehicle collision data from video processing pipelines.

## Tools Overview

The tools implement the specifications from `tools-plan.md` and provide a complete workflow for accident analysis:

### 1. `load_detections(track_ids, frame_range=None, fields=None, require_pairing=True, fps_hint=None)`

Loads detection data for specified track IDs from the detections.jsonl file.

**Parameters:**

- `track_ids`: List of track IDs to load (e.g., [7, 14])
- `frame_range`: Optional (start_frame, end_frame) tuple to restrict loading
- `fields`: Optional list of fields to include (controls payload size)
- `require_pairing`: If True, only return frames where all track_ids appear
- `fps_hint`: Optional FPS hint for metadata
- `detections_file`: Path to the detections.jsonl file

**Returns:**

- Dictionary with 'records' (paired detection records) and 'metadata' (summary stats)

### 2. `compute_pair_metrics(pairs, iou_threshold=0.01, vehicle_length_m=4.5, include_headings=False)`

Computes collision metrics for paired detections.

**Parameters:**

- `pairs`: List of paired detection records (from load_detections)
- `iou_threshold`: IoU threshold for collision detection
- `vehicle_length_m`: Assumed vehicle length for distance calculations
- `include_headings`: Whether to compute heading differences

**Returns:**

- List of MetricRow objects with enriched data (IoU, distances, speeds, flags)

### 3. `trace_impact_window(metric_rows, iou_threshold=0.01, distance_threshold_m=5.0, persistence_frames=3)`

Traces the impact window to detect collision events.

**Parameters:**

- `metric_rows`: List of MetricRow objects
- `iou_threshold`: IoU threshold for collision detection
- `distance_threshold_m`: Distance threshold in meters
- `persistence_frames`: Number of frames to persist overlap

**Returns:**

- Dictionary with collision detection results, impact frames, and diagnostic notes

### 4. `build_timeline(metric_rows, impact_summary, padding_frames=10, stages=None)`

Builds a structured timeline of events.

**Parameters:**

- `metric_rows`: List of MetricRow objects
- `impact_summary`: Output from trace_impact_window
- `padding_frames`: Number of frames to include before/after impact
- `stages`: Custom stage labels (defaults to approach, first_contact, peak_overlap, separation)

**Returns:**

- Dictionary with timeline entries and summary table

### 5. `report_assumptions(metric_rows, metadata, warn_if_missing=("world_coords","speed_mph"))`

Reports data quality issues and assumptions.

**Parameters:**

- `metric_rows`: List of MetricRow objects
- `metadata`: Metadata from load_detections
- `warn_if_missing`: Fields to warn about if missing

**Returns:**

- List of assumption/gap descriptions

## Data Structures

### Detection

Represents a single detection record with fields:

- `video_id`, `frame`, `time`, `track_id`
- `bbox_xyxy`, `center`, `speed_mph`, `world_coords`
- `conf`, `class_name`, etc.

### MetricRow

Enriched detection data with computed metrics:

- `frame`, `timestamp`, `track_ids`
- `iou`, `pixel_center_distance_px`, `world_distance_m`
- `relative_speed_mps`, `heading_diff_deg`
- Boolean flags: `iou_exceeds_threshold`, `distance_below_threshold`, `collision_candidate`

## Usage Examples

### Basic Workflow

```python
from tools import load_detections, compute_pair_metrics, trace_impact_window, build_timeline, report_assumptions

# 1. Load data
result = load_detections(track_ids=[7, 14], frame_range=(2, 15))
records = result['records']
metadata = result['metadata']

# 2. Compute metrics
metric_rows = compute_pair_metrics(pairs=records, iou_threshold=0.01)

# 3. Analyze impact
impact_summary = trace_impact_window(metric_rows, iou_threshold=0.01, distance_threshold_m=5.0)

# 4. Build timeline
timeline_result = build_timeline(metric_rows, impact_summary, padding_frames=5)

# 5. Report quality
assumptions = report_assumptions(metric_rows, metadata)
```

### Testing

Run the test script to verify functionality:

```bash
python test_tools.py
```

Run the example workflow:

```bash
python example_usage.py
```

## Integration with LLM Agents

These tools are designed to be called by LLM agents as function calls. The tools provide:

1. **Structured data loading** with flexible filtering
2. **Collision detection** with configurable thresholds
3. **Event timeline generation** for narrative construction
4. **Data quality reporting** for assumption validation
5. **Consistent data structures** for downstream processing

The tools handle edge cases gracefully and provide diagnostic information to help LLMs understand data limitations and make informed decisions about accident analysis.

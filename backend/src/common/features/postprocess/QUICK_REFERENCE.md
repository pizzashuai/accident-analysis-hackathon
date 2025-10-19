# Accident Analysis Agent - Quick Reference

## 🚀 Quick Start

### Local Testing (3 commands)

```bash
cd backend/src/common/features/postprocess

# Basic analysis
python main.py --track-ids 7 14

# Save report
python main.py --track-ids 7 14 --output report.json

# Run examples
python main.py --examples
```

### Python Usage (5 lines)

```python
from postprocess import AccidentAnalysisAgent, AgentConfig

config = AgentConfig(track_ids=[7, 14], detections_file="path/to/detections.jsonl")
agent = AccidentAnalysisAgent(config)
report = agent.analyze()
```

## 📋 Tools Overview

| Tool                   | Purpose                          | Key Output         |
| ---------------------- | -------------------------------- | ------------------ |
| `load_detections`      | Load paired track data           | Records + metadata |
| `compute_pair_metrics` | Calculate IoU, distances, speeds | MetricRow objects  |
| `trace_impact_window`  | Detect collision/near-miss       | Impact summary     |
| `build_timeline`       | Generate event sequence          | Timeline + table   |
| `report_assumptions`   | Data quality check               | Warnings list      |

## ⚙️ Key Parameters

### Detection Thresholds

```python
AgentConfig(
    iou_threshold=0.01,         # Lower = more sensitive (0.001-0.05)
    distance_threshold_m=5.0,   # Larger = wider detection (2.0-10.0)
    persistence_frames=3         # Higher = stricter (1-7)
)
```

### Common Scenarios

**Strict Detection** (High confidence only)

```python
iou_threshold=0.05, distance_threshold_m=2.0, persistence_frames=5
```

**Sensitive Detection** (Catch glancing contacts)

```python
iou_threshold=0.001, distance_threshold_m=10.0, persistence_frames=1
```

**Balanced** (Default)

```python
iou_threshold=0.01, distance_threshold_m=5.0, persistence_frames=3
```

## 📊 Report Structure

```python
report = {
    "success": bool,
    "collision_detected": bool,
    "narrative_summary": str,           # Human-readable summary
    "timeline": [str],                  # Event citations
    "summary_table": [dict],            # Tabular data
    "impact_analysis": {
        "type": "COLLISION" | "NEAR-MISS",
        "severity_indicators": {...},   # IoU, distance, duration
        "key_frames": {...}             # first_contact, peak, separation
    },
    "data_quality": {
        "assumptions": [str],           # Data issues/warnings
        "total_frames_analyzed": int
    },
    "metrics_summary": {...}
}
```

## 🔗 API Integration

### FastAPI Endpoint

```python
from fastapi import FastAPI
from postprocess.integration_example import create_fastapi_endpoint

app = FastAPI()
app.include_router(create_fastapi_endpoint())

# POST /analysis/collision
# Body: {"project_id": 1, "track_ids": [7, 14]}
```

### Celery Task

```python
from postprocess.integration_example import create_celery_task

analyze_task = create_celery_task()
result = analyze_task.apply_async(
    args=[project_id, [7, 14], detections_file]
)
```

### Async Function

```python
from postprocess.integration_example import analyze_collision_async

report = await analyze_collision_async(
    track_ids=[7, 14],
    detections_file="path/to/file"
)
```

## 🌩️ AWS Deployment

### 1. Generate Schema

```bash
python agent_aws.py
# Creates: openapi_schema.json
```

### 2. Lambda Handler

```python
from postprocess.agent_aws import BedrockAgentHandler

handler = BedrockAgentHandler(detections_file="/mnt/efs/detections.jsonl")

def lambda_handler(event, context):
    return handler.lambda_handler(event, context)
```

### 3. Bedrock Configuration

- Upload `openapi_schema.json` to action group
- Point to Lambda function
- Use system prompt from `agent_core.py`

## 🔍 Troubleshooting

### Issue: No collision detected

**Fix**: Lower thresholds

```python
iou_threshold=0.001
distance_threshold_m=10.0
persistence_frames=1
```

### Issue: False positives

**Fix**: Raise thresholds

```python
iou_threshold=0.05
distance_threshold_m=2.0
persistence_frames=5
```

### Issue: Missing speed data

**Check**:

- Report warnings: `report['data_quality']['assumptions']`
- Homography configuration
- Frame sampling rate

### Issue: Performance slow

**Fix**:

- Use `frame_range` parameter
- Reduce `padding_frames`
- Process in batches

## 📝 CLI Reference

```bash
# Basic
python main.py --track-ids 7 14

# Frame range
python main.py --track-ids 7 14 --frame-range 10 50

# Custom thresholds
python main.py --track-ids 7 14 \
  --iou-threshold 0.02 \
  --distance-threshold 3.0 \
  --persistence-frames 5

# Output to file
python main.py --track-ids 7 14 --output report.json

# Custom detections file
python main.py --track-ids 7 14 \
  --detections-file /path/to/detections.jsonl

# Run examples
python main.py --examples

# Help
python main.py --help
```

## 🎯 Common Use Cases

### 1. Single Collision Analysis

```python
config = AgentConfig(track_ids=[7, 14], detections_file="file.jsonl")
agent = AccidentAnalysisAgent(config)
report = agent.analyze()

if report['collision_detected']:
    print(f"Collision at frame {report['impact_analysis']['key_frames']['first_contact']}")
```

### 2. Batch Analysis

```python
from postprocess.integration_example import analyze_multiple_collisions

collision_pairs = [(7, 14), (1, 2), (3, 5)]
reports = analyze_multiple_collisions(collision_pairs, "file.jsonl")
```

### 3. Frontend-Ready Response

```python
from postprocess.integration_example import format_for_frontend

report = agent.analyze()
frontend_data = format_for_frontend(report)
# Returns simplified structure with key metrics
```

### 4. Save to Database

```python
from postprocess.integration_example import AnalysisResult

report = agent.analyze()
AnalysisResult.save_to_database(project_id, report, db_session)
```

## 📚 Documentation

- **Full Guide**: `AGENT_README.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Tools API**: `README.md`
- **Examples**: `example_usage.py`, `integration_example.py`
- **Tests**: `test_tools.py`

## 🏗️ Project Structure

```
postprocess/
├── tools.py                 # Core analysis functions
├── agent_tools.py          # AWS schemas + handler
├── agent_core.py           # Orchestration logic
├── agent_aws.py            # Bedrock integration
├── main.py                 # CLI testing
├── integration_example.py  # Backend patterns
└── *.md                    # Documentation
```

## 🔑 Key Concepts

**IoU (Intersection over Union)**: Overlap between bounding boxes (0-1)

- 0.0 = No overlap
- 0.01 = Minimal overlap (typical threshold)
- 0.05+ = Significant overlap

**World Distance**: GPS-based distance in meters (Haversine formula)

- Requires homography mapping
- Typical collision: < 5m

**Persistence Frames**: Consecutive frames meeting collision criteria

- Reduces false positives from single-frame noise
- Typical: 3 frames

**Timeline Stages**:

1. **Approach** - Vehicles closing distance
2. **First Contact** - Initial collision detection
3. **Peak Overlap** - Maximum IoU/minimum distance
4. **Separation** - Vehicles moving apart

## 💡 Pro Tips

1. **Start with defaults** - Adjust only if needed
2. **Check data quality** - Review `assumptions` before trusting results
3. **Use frame ranges** - Focus on relevant portions for speed
4. **Save reports** - JSON output for reproducibility
5. **Test locally first** - Use `main.py` before AWS deployment

## 📞 Support

- Check documentation in `AGENT_README.md`
- Review examples in `integration_example.py`
- Run tests with `python test_tools.py`
- Use `python main.py --help` for CLI options

---

**Version**: 1.0.0 | **Updated**: October 2025

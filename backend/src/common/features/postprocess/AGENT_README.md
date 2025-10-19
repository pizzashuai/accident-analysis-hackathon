# Accident Analysis Agent

This directory contains a complete AWS Bedrock Agent integration for automated accident reconstruction analysis from video detection data.

## Overview

The agent follows the blueprint specified in `/llm-hints/agent-plan.md` and provides:

1. **Analytical Tools** - Functions for loading, processing, and analyzing collision data
2. **Agent Core** - Orchestration logic that follows the collision analysis workflow
3. **AWS Integration** - Bedrock Agent handlers and schemas for cloud deployment
4. **Local Testing** - Standalone execution for development and validation

## Architecture

```
postprocess/
├── tools.py              # Core analysis functions (load, compute, trace, build, report)
├── agent_tools.py        # AWS agent tool schemas and handler
├── agent_core.py         # Agent orchestration logic
├── agent_aws.py          # AWS Bedrock integration
├── main.py               # Local testing entry point
├── example_usage.py      # Simple workflow examples
└── test_tools.py         # Unit tests
```

## Quick Start - Local Testing

### Basic Analysis

Analyze a collision between two tracks:

```bash
cd backend/src/common/features/postprocess
python main.py --track-ids 7 14
```

### With Frame Range

Analyze specific frames:

```bash
python main.py --track-ids 7 14 --frame-range 2 30
```

### Adjust Detection Thresholds

```bash
python main.py --track-ids 7 14 \
  --iou-threshold 0.02 \
  --distance-threshold 3.0 \
  --persistence-frames 5
```

### Save Report

```bash
python main.py --track-ids 7 14 --output collision_report.json
```

### Run Example Scenarios

```bash
python main.py --examples
```

## Agent Workflow

The agent follows this orchestration flow from `agent-plan.md`:

1. **Load Detections** (`load_detections`)

   - Load paired detection records for specified track IDs
   - Check for missing frames and data gaps
   - Return metadata about FPS and time range

2. **Compute Metrics** (`compute_pair_metrics`)

   - Calculate IoU (Intersection over Union) between bounding boxes
   - Compute world distance using GPS coordinates
   - Calculate relative speeds
   - Flag collision candidates

3. **Trace Impact Window** (`trace_impact_window`)

   - Identify frames meeting collision criteria
   - Determine if collision occurred (vs. near-miss)
   - Find first contact, peak overlap, and separation frames
   - Generate diagnostic notes

4. **Build Timeline** (`build_timeline`)

   - Create structured timeline with key events
   - Generate narrative for each stage
   - Format with frame/timestamp/metrics citations

5. **Report Assumptions** (`report_assumptions`)

   - Identify missing data (speeds, coordinates)
   - Flag low-confidence detections
   - Note frame gaps and data quality issues

6. **Generate Final Report**
   - Narrative summary of collision/near-miss
   - Chronological bullet timeline with citations
   - Impact analysis with severity indicators
   - Data quality section with assumptions

## Output Format

The agent produces a comprehensive JSON report with:

```json
{
  "success": true,
  "collision_detected": true,
  "narrative_summary": "COLLISION DETECTED between vehicles...",
  "timeline": [
    "[APPROACH] Frame 2 (t=0.067s): Vehicles approaching with 40.1m separation | IoU=0.000, Distance=40.1m",
    "[FIRST_CONTACT] Frame 8 (t=0.267s): First contact detected with IoU 0.023 | IoU=0.023, Distance=3.2m",
    ...
  ],
  "summary_table": [...],
  "impact_analysis": {
    "type": "COLLISION",
    "severity_indicators": {
      "max_iou": 0.156,
      "min_distance_m": 0.8,
      "impact_duration_frames": 12
    },
    "key_frames": {
      "first_contact": 8,
      "last_overlap": 20,
      "closest_approach": 15
    }
  },
  "data_quality": {
    "total_frames_analyzed": 45,
    "assumptions": [...]
  }
}
```

## AWS Bedrock Integration

### Setup

1. **Generate OpenAPI Schema**

```bash
python agent_aws.py
```

This creates `openapi_schema.json` for Bedrock configuration.

2. **Create Lambda Function**

```python
from postprocess.agent_aws import BedrockAgentHandler

handler = BedrockAgentHandler(detections_file="/path/to/detections.jsonl")

def lambda_handler(event, context):
    return handler.lambda_handler(event, context)
```

3. **Configure Bedrock Agent**

- Create agent in AWS Bedrock console
- Add action group with the OpenAPI schema
- Point to Lambda function
- Configure IAM roles and permissions

### Tool Schemas

The agent exposes 5 tools to AWS Bedrock:

1. `load_detections` - Load detection data for track IDs
2. `compute_pair_metrics` - Calculate collision metrics
3. `trace_impact_window` - Detect collision events
4. `build_timeline` - Generate event timeline
5. `report_assumptions` - Report data quality issues

### Agent Prompt

Use this system prompt for AWS Bedrock:

```
You are an accident reconstruction analyst specializing in vehicle collision analysis.

Your goal is to analyze detection data from video processing to determine if a collision
occurred and produce a comprehensive timeline.

CRITICAL REQUIREMENTS:
- Every event citation must include frame number, timestamp, and key numeric metrics
- Verify units: distances in meters, speeds in m/s or mph, time in seconds
- Request follow-up tool calls whenever more data is needed
- Surface any missing metadata as warnings

WORKFLOW:
1. Load detections with specified track IDs
2. Check metadata and record assumptions if data is missing
3. Compute collision metrics
4. Trace impact window to identify collision/near-miss
5. Build structured timeline
6. Generate comprehensive report with citations
```

## Development

### Running Tests

```bash
# Run unit tests
python test_tools.py

# Run example workflow
python example_usage.py

# Test agent locally
python main.py --track-ids 7 14
```

### Adding Custom Analysis

Extend the agent by:

1. Adding new tools to `tools.py`
2. Creating tool schemas in `agent_tools.py`
3. Updating orchestration in `agent_core.py`
4. Regenerating OpenAPI schema with `agent_aws.py`

## Configuration

### AgentConfig

```python
from agent_core import AgentConfig, AccidentAnalysisAgent

config = AgentConfig(
    track_ids=[7, 14],              # Required: Track IDs to analyze
    frame_range=(2, 30),             # Optional: Limit frame range
    iou_threshold=0.01,              # IoU threshold for collision
    distance_threshold_m=5.0,        # Distance threshold in meters
    persistence_frames=3,            # Frames to persist overlap
    padding_frames=10,               # Context frames around impact
    detections_file="path/to/file"   # Path to detections.jsonl
)

agent = AccidentAnalysisAgent(config)
report = agent.analyze()
```

## Best Practices

### Threshold Selection

- **IoU Threshold** (default: 0.01)

  - Lower values (0.001-0.01): Detect glancing contacts
  - Higher values (0.02-0.05): Require significant overlap

- **Distance Threshold** (default: 5.0m)

  - Adjust based on vehicle size and GPS accuracy
  - Consider sensor error margins

- **Persistence Frames** (default: 3)
  - Higher values reduce false positives
  - Lower values catch brief contacts

### Data Quality

Always check the `data_quality` section of reports:

- Missing world coordinates → distance calculations unavailable
- Missing speeds → relative velocity analysis limited
- Frame gaps → timeline may have discontinuities
- Low confidence detections → bounding boxes less reliable

## Examples

### Example 1: Basic Collision Detection

```python
from agent_core import AccidentAnalysisAgent, AgentConfig

config = AgentConfig(
    track_ids=[7, 14],
    detections_file="../process_video/detections.jsonl"
)

agent = AccidentAnalysisAgent(config)
report = agent.analyze()

if report['collision_detected']:
    print(f"Collision at frame {report['impact_analysis']['key_frames']['first_contact']}")
else:
    print(f"Near-miss: closest {report['impact_analysis']['closest_approach']['distance_m']:.1f}m")
```

### Example 2: Strict Analysis

```python
# Require strong evidence of collision
config = AgentConfig(
    track_ids=[7, 14],
    iou_threshold=0.05,           # Significant overlap required
    distance_threshold_m=2.0,      # Very close approach
    persistence_frames=5,          # Sustained contact
    detections_file="../process_video/detections.jsonl"
)

agent = AccidentAnalysisAgent(config)
report = agent.analyze()
```

### Example 3: Frame-by-Frame Analysis

```python
# Analyze collision progression in detail
config = AgentConfig(
    track_ids=[7, 14],
    frame_range=(5, 25),    # Focus on impact window
    padding_frames=2,        # Minimal context
    detections_file="../process_video/detections.jsonl"
)

agent = AccidentAnalysisAgent(config)
report = agent.analyze()

# Examine each timeline event
for event in report['timeline']:
    print(event)
```

## Troubleshooting

### No collision detected when expected

- Lower `iou_threshold` (try 0.001)
- Increase `distance_threshold_m` (try 10.0)
- Reduce `persistence_frames` (try 1-2)
- Check `data_quality.assumptions` for missing data

### False positive collisions

- Increase `iou_threshold` (try 0.02-0.05)
- Decrease `distance_threshold_m` (try 2.0-3.0)
- Increase `persistence_frames` (try 5-7)

### Missing data warnings

- Speed data: Requires sufficient frames for velocity calculation
- World coordinates: Requires homography mapping setup
- Frame gaps: Check video processing for dropped frames

## License

See project LICENSE file.

## References

- Agent Blueprint: `/llm-hints/agent-plan.md`
- Tools Plan: `/llm-hints/tools-plan.md`
- Tools Documentation: `README.md`
- Example Usage: `example_usage.py`

# LLM-Powered Accident Analysis Agent

This module provides a flexible, LLM-driven accident analysis system that uses AWS Bedrock Claude to analyze vehicle collision data and generate dynamic reports based on the characteristics of the data.

## Key Features

- **LLM-Driven Analysis**: Uses AWS Bedrock Claude to intelligently analyze accident data
- **Flexible Output Formats**: The LLM decides the report structure based on data characteristics
- **No Hardcoded Formats**: Removed all hardcoded report templates and narrative structures
- **Tool-Based Architecture**: Uses specialized tools for data analysis
- **Comprehensive Analysis**: Analyzes IoU, distances, speeds, timelines, and data quality

## Files

- `llm_agent.py` - Main LLM-powered agent implementation
- `main.py` - Command-line interface for running analysis
- `test_agent.py` - Test suite to verify functionality
- `agent_tools.py` - Tool handler for data analysis functions
- `tools.py` - Core analysis functions

## Usage

### Basic Analysis

```bash
python main.py --track-ids 7 14
```

### With Custom Parameters

```bash
python main.py --track-ids 7 14 --frame-range 2 30 --iou-threshold 0.02 --distance-threshold 3.0
```

### Save Report to File

```bash
python main.py --track-ids 7 14 --output report.json
```

### Test Functionality

```bash
python test_agent.py
```

## Configuration

The agent uses AWS Bedrock Claude for analysis. Configure AWS credentials:

```bash
aws configure
# or set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

## What Changed

### Removed Hardcoded Formats

- ❌ Fixed report templates and narrative structures
- ❌ Predefined report sections (EXECUTIVE SUMMARY, EVENT TIMELINE, etc.)
- ❌ Hardcoded collision analysis report JSON
- ❌ Static timeline formatting

### Added LLM-Driven Analysis

- ✅ LLM decides output format based on data characteristics
- ✅ Flexible report generation that adapts to data complexity
- ✅ Dynamic analysis workflow
- ✅ Intelligent tool selection and parameter adjustment

## Analysis Tools

1. **load_detections** - Load detection data for specified track IDs
2. **compute_pair_metrics** - Calculate collision metrics (IoU, distances, speeds)
3. **trace_impact_window** - Detect collision events and impact windows
4. **build_timeline** - Generate structured event timeline
5. **report_assumptions** - Identify data quality issues and assumptions

## Example Output

The LLM generates comprehensive reports that include:

- Executive summary based on actual data findings
- Event timeline with specific frame citations
- Impact analysis with severity indicators
- Data quality assessment
- Conclusions with supporting evidence

The format and detail level adapt automatically based on the complexity and characteristics of the input data.

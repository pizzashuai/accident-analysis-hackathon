<!-- 143ff074-1732-49dc-89e4-234445b63f3d 5c0d7525-e45a-4573-af22-4a2ca85a960d -->
# LLM Streaming API Implementation Plan

## Overview

Implement a streaming API that allows the frontend to receive real-time updates from the LLM accident analysis agent, including thinking/reasoning steps, tool calls, and the final report. The LLM processing will run in a Celery background task, and the frontend will connect via Server-Sent Events (SSE) to receive updates.

## Architecture

### Flow

1. Frontend calls `/api/v1/projects/{project_id}/llm-analysis/start` with `run_id`
2. API validates project ownership and retrieves the filtered JSONL artifact for the run
3. API creates a unique `analysis_id` and starts a Celery task
4. API returns `analysis_id` to frontend
5. Frontend connects to SSE endpoint `/api/v1/projects/{project_id}/llm-analysis/{analysis_id}/stream`
6. Celery task runs LLM analysis, publishing events to Redis
7. SSE endpoint streams events to frontend in real-time
8. When complete, final report is streamed

### Event Types (ChatGPT-style)

- `thinking_start`: LLM begins reasoning (show thinking UI)
- `thinking_content`: Streaming thinking/reasoning text
- `thinking_end`: Thinking phase complete
- `tool_call_start`: Tool execution begins
- `tool_call_result`: Tool execution result
- `report_start`: Final report generation begins
- `report_content`: Streaming report text (markdown)
- `report_end`: Analysis complete
- `error`: Error occurred

## Implementation Steps

### 1. Create Redis-based Event Publishing System

**File**: `src/common/features/postprocess/event_publisher.py` (new)

- Create `LLMEventPublisher` class that publishes events to Redis pub/sub
- Events include: thinking updates, tool calls, tool results, report chunks
- Use Redis channels keyed by `analysis_id`

### 2. Modify LLM Agent to Publish Events

**File**: `src/common/features/postprocess/llm_agent.py`

- Add `event_publisher` parameter to `LLMAccidentAnalysisAgent.__init__()`
- Modify `_call_bedrock_with_tools()` to publish thinking events
- Modify `_process_tool_use()` to publish tool call events
- Modify `_extract_final_report()` to stream report in chunks
- Publish events at each step: model calls, tool executions, report generation

### 3. Create Celery Task for LLM Analysis

**File**: `src/worker/celery_app/tasks.py`

- Add `analyze_accident_llm_task(analysis_id, project_id, run_id, detections_file_path)`
- Download filtered JSONL artifact from S3 to temp file
- Initialize `LLMAccidentAnalysisAgent` with event publisher
- Extract track IDs from filtered JSONL metadata
- Run analysis and publish events to Redis
- Handle errors and publish error events

### 4. Create SSE Streaming Endpoint

**File**: `src/api/routes/llm_analysis_route.py` (new)

- `POST /projects/{project_id}/llm-analysis/start`: Start analysis
  - Validate project ownership
  - Get latest completed processing run for project
  - Find filtered JSONL artifact (kind="jsonl_detections" with "filtered_track_ids" in meta)
  - Generate unique `analysis_id`
  - Start Celery task with analysis_id
  - Return `{analysis_id, status: "started"}`

- `GET /projects/{project_id}/llm-analysis/{analysis_id}/stream`: SSE stream
  - Validate project ownership
  - Subscribe to Redis channel for `analysis_id`
  - Stream events as SSE format:
    ```
    event: thinking_start
    data: {"message": "Analyzing collision data..."}
    
    event: thinking_content
    data: {"content": "I need to load the detection data first..."}
    
    event: tool_call_start
    data: {"tool": "load_detections", "input": {...}}
    
    event: tool_call_result
    data: {"tool": "load_detections", "result": {...}}
    
    event: report_start
    data: {"message": "Generating final report..."}
    
    event: report_content
    data: {"content": "## Accident Analysis Report\n\n..."}
    
    event: report_end
    data: {"message": "Analysis complete"}
    ```


### 5. Register New Route in Main API

**File**: `src/api/main.py`

- Import `llm_analysis_route`
- Add `api_router.include_router(llm_analysis_route.router)`

### 6. Update Dependencies

**File**: `pyproject.toml`

- Ensure `redis` package is included (likely already present for Celery)
- Add `sse-starlette` for SSE support in FastAPI

## Key Files to Create/Modify

### New Files

- `src/common/features/postprocess/event_publisher.py`: Event publishing to Redis
- `src/api/routes/llm_analysis_route.py`: SSE streaming endpoints

### Modified Files

- `src/common/features/postprocess/llm_agent.py`: Add event publishing
- `src/worker/celery_app/tasks.py`: Add LLM analysis Celery task
- `src/api/main.py`: Register new route

## Frontend Integration Notes

The frontend should:

1. Call start endpoint to get `analysis_id`
2. Connect to SSE stream endpoint
3. Display thinking events in a collapsible "thinking" UI (like ChatGPT)
4. Display tool calls as badges or status indicators
5. Display report content in markdown with streaming effect
6. Handle errors gracefully

## Technical Details

### Redis Channel Naming

- Channel: `llm_analysis:{analysis_id}`
- TTL: 1 hour (events expire after analysis)

### Event Format

```json
{
  "type": "thinking_content" | "tool_call_start" | "report_content" | ...,
  "data": {...},
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Handling

- Publish error events if LLM fails
- SSE connection timeout after 5 minutes of inactivity
- Celery task timeout: 10 minutes

### To-dos

- [ ] Create Redis-based event publisher for LLM events
- [ ] Modify LLM agent to publish events during analysis
- [ ] Create Celery task for LLM analysis with event publishing
- [ ] Create SSE streaming endpoint for real-time event delivery
- [ ] Create start analysis endpoint to trigger Celery task
- [ ] Register new LLM analysis routes in main API
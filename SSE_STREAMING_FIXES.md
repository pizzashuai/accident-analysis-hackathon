# SSE Streaming Fixes

## Issues Fixed

### Issue 1: Frontend Not Displaying Streamed Content (✅ FIXED)

**Problem**: Backend was sending custom SSE event types (like `thinking_start`, `report_content`, etc.), but the frontend was only listening to the generic `message` event type.

**Root Cause**: EventSource's `onmessage` handler only catches events without a custom type or events explicitly typed as "message". All the custom event types from the backend were being ignored.

**Solution**: Modified `useSSE.ts` to register event listeners for all custom event types:

- `connected`
- `thinking_start`, `thinking_content`, `thinking_end`
- `tool_call_start`, `tool_call_result`
- `report_start`, `report_content`, `report_end`
- `collision_detected`
- `iteration_update`
- `model_switch`
- `error`

### Issue 2: SSE Connection Error on Normal Close (✅ FIXED)

**Problem**: When the backend completed analysis and closed the connection (after sending `report_end`), the frontend treated this as an error and logged "SSE connection error".

**Root Cause**: EventSource automatically fires an `onerror` event when the connection closes, even if it's a normal/expected closure. The frontend was treating all `onerror` events as errors.

**Solution**:

1. Added `expectedCloseRef` to track when a terminal event (`report_end` or `error`) has been received
2. Modified error handler to only report errors if the close was unexpected
3. Added graceful close with 100ms delay after terminal events to ensure all events are processed
4. Added `onClose` callback to notify when connection closes gracefully

### Issue 3: Parse Error on Empty/Undefined Event Data (✅ FIXED)

**Problem**: Frontend was trying to parse `event.data` as JSON even when it was `undefined` or empty, causing "JSON.parse: unexpected character" errors. Error message: "Failed to parse SSE error event: SyntaxError: JSON.parse: unexpected character at line 1 column 1 of the JSON data, Raw data: undefined"

**Root Cause**:

1. EventSource can fire events with `undefined` or empty `data` during connection lifecycle
2. Frontend was not validating that `event.data` exists and is a string before calling `JSON.parse()`
3. Spurious error events during normal connection close were being processed unnecessarily
4. Backend was closing connection immediately after sending terminal events, causing race conditions

**Solution**:

1. **Frontend**: Added validation to check if `event.data` exists, is a string, and is not empty before parsing
2. **Frontend**: Added logic to skip error events when connection is already marked for expected close
3. **Backend**: Added 100ms delay after terminal events to ensure all events are flushed before connection closes

## Files Modified

### Frontend Changes

1. **`/frontend/app/hooks/useSSE.ts`**

   - Added `expectedCloseRef` to track expected connection closes
   - Added custom event listeners for all backend event types
   - Modified error handler to differentiate between expected and unexpected closes
   - Added graceful close logic for terminal events
   - **NEW (Issue #3):** Added data validation to check if `event.data` exists and is a valid string before parsing
   - **NEW (Issue #3):** Added logic to skip spurious error events during expected connection close

2. **`/frontend/app/hooks/useLLMAnalysis.ts`**
   - Added `onClose` callback to log graceful connection closes
   - Removed redundant "test" log message

### Backend Changes

1. **`/backend/src/api/routes/llm_analysis_route.py`**
   - **NEW (Issue #3):** Added 100ms delay before closing connection after terminal events (`report_end` or `error`)
   - This ensures all SSE events are flushed to the client before the connection closes

## Testing

To verify the fixes:

1. **Start the backend** (API and worker):

   ```bash
   cd backend
   ./scripts/run-api.sh  # In terminal 1
   ./scripts/run-worker.sh  # In terminal 2
   ```

2. **Start the frontend**:

   ```bash
   cd frontend
   pnpm dev
   ```

3. **Test the LLM Analysis**:

   - Go to a project that has completed video processing
   - Click on the "AI Accident Analysis" section
   - Click "Start Analysis"
   - You should now see:
     - ✅ "Thinking Process" content streaming in real-time
     - ✅ "Tool Usage" appearing as tools are called
     - ✅ "Analysis Report" streaming in chunks
     - ✅ Smooth completion without errors
     - ✅ No "SSE connection error" in console

4. **Check Browser Console**:

   - Should see logs like:
     ```
     SSE connection opened for LLM analysis
     SSE thinking_start event received: ...
     SSE report_content event received: ...
     SSE report_end event received: ...
     Closing SSE connection after report_end event
     SSE connection closed gracefully
     ```
   - Should NOT see any errors about "SSE connection error"

5. **Check Backend Logs**:
   - API logs should show events being streamed:
     ```
     INFO: Streaming event thinking_start with data: ...
     INFO: Streaming event report_content with data: ...
     INFO: Streaming event report_end with data: ...
     INFO: Closing SSE stream for analysis <id>
     ```
   - Worker logs should show events being published:
     ```
     INFO: Successfully published event thinking_start
     INFO: Successfully published event report_content
     INFO: Successfully published event report_end
     ```

## Event Flow

```
Backend Worker → Redis Pub/Sub → API SSE Stream → Frontend EventSource → UI Update
     ↓                 ↓                ↓                    ↓               ↓
[LLM Agent]    [Redis Channel]  [event_generator]   [useSSE hook]   [React State]
 publishes      llm_analysis:    yields formatted    receives        updates UI
 event data     {analysis_id}    SSE events         custom events   components
```

## Debugging Tips

If issues persist:

1. **Check Redis connection**:

   ```bash
   redis-cli ping  # Should return PONG
   ```

2. **Monitor Redis events**:

   ```bash
   redis-cli monitor  # Watch all Redis commands
   ```

3. **Check API SSE endpoint**:

   ```bash
   # Get auth token from browser localStorage
   curl -N "http://localhost:8000/api/v1/llm-analysis/projects/{project_id}/stream/{analysis_id}?token={token}"
   ```

4. **Check browser DevTools**:
   - Network tab → Look for EventSource connection
   - Should show "text/event-stream" content type
   - Should see events streaming in Messages tab

## Key Learnings

1. **Custom SSE Event Types**: When using custom event types in SSE (e.g., `event: thinking_start`), you must register explicit event listeners with `addEventListener(eventType, handler)`. The generic `onmessage` handler only catches events without a type or with type "message".

2. **EventSource Connection Lifecycle**: EventSource fires `onerror` on ANY connection close, including normal/expected closes. Must track state to differentiate between error conditions and graceful shutdowns.

3. **SSE Format**: Server-Sent Events must be formatted as:
   ```
   event: custom_type\n
   data: {"key": "value"}\n\n
   ```
   The double newline (`\n\n`) is critical for message boundaries.

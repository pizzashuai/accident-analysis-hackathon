<!-- 033629a4-1fcd-4b52-9883-b2ab8325f2bf 7adc8dd4-aa27-4f6e-a312-241436e961d3 -->
# Phase 2 - Homography Capture & Solving Implementation

## Overview

Enable users to capture point correspondence pairs between CCTV video screenshots and map coordinates, persist them in the database, and compute homography transformation matrices for speed calculation and spatial analysis. This implementation reuses the existing homography solver from `process-video/src/estimate_distance.py`.

## Backend: Database Models

Create new SQLAlchemy models in `backend/src/common/database/models/`:

### 1. `homography_session_table.py`

```python
- id: UUID (PK)
- project_id: UUID (FK → project.id, CASCADE)
- screenshot_asset_id: UUID (FK → media_asset.id, nullable)
- status: String ('draft', 'ready_to_solve', 'solved', 'error')
- created_at: DateTime
- solved_at: DateTime (nullable)
```

### 2. `homography_pair_table.py`

```python
- id: UUID (PK)
- session_id: UUID (FK → homography_session.id, CASCADE)
- image_x_norm: Float (0-1 normalized)
- image_y_norm: Float (0-1 normalized)
- map_lat: Float
- map_lng: Float
- order_idx: Integer (for display ordering)
```

### 3. `homography_model_table.py`

```python
- id: UUID (PK)
- session_id: UUID (FK → homography_session.id, CASCADE)
- matrix_data: JSONB (3x3 matrix as nested array)
- reprojection_error: Float (nullable)
- created_at: DateTime
- meta: JSONB (additional metrics)
```

**Update** `backend/src/common/database/models/__init__.py` to export new models.

**Generate Alembic migration** with:

```bash
alembic revision --autogenerate -m "add_homography_tables"
```

## Backend: Reuse Existing Homography Solver

The homography solver already exists in `backend/src/common/features/process-video/src/estimate_distance.py`:

**Key classes**:

- `DistanceEstimator` - loads JSON and calculates homography with `cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)`
- `ImagePoint`, `GeoPoint`, `PointPair`, `HomographyData` - data structures matching frontend format

**Reuse strategy**: Import and adapt existing code rather than reimplementing.

## Backend: Pydantic Schemas & CRUD

Create `backend/src/common/features/homography/` directory:

### `schemas.py`

```python
- HomographyPairCreate (image_x_norm, image_y_norm, map_lat, map_lng)
- HomographyPairPublic (id, session_id, coordinates, order_idx)
- HomographySessionCreate (project_id, screenshot_asset_id optional)
- HomographySessionUpdate (status, pairs list)
- HomographySessionPublic (id, project_id, status, pairs, model, created_at, solved_at)
- HomographyModelPublic (id, session_id, matrix_data, reprojection_error, created_at)
- HomographySolveResponse (success, model, error message optional)
```

### `crud.py`

Key functions:

- `create_session(db, project_id) → HomographySession`
- `get_session(db, session_id) → HomographySession | None`
- `get_or_create_session_for_project(db, project_id) → HomographySession`
- `add_pair(db, session_id, pair_data) → HomographyPair`
- `update_pairs(db, session_id, pairs_list)` - replace all pairs
- `delete_pair(db, pair_id)`
- `solve_homography(db, session_id) → HomographyModel` - calls solver

### `solver.py`

Wrapper around existing `estimate_distance.py`:

```python
from src.common.features.process_video.src.estimate_distance import DistanceEstimator
import numpy as np
import cv2

def solve_homography_from_pairs(pairs: List[HomographyPair]) -> dict:
    # Reuses existing DistanceEstimator logic:
    # 1. Convert DB pairs to normalized coordinates
    # 2. Build src_points (xNorm, yNorm) and dst_points (lng, lat)
    # 3. Call cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    # 4. Calculate reprojection error for each point
    # 5. Return {matrix: [[...]], reprojection_error: float, inlier_count: int}
```

## Backend: API Routes

Extend `backend/src/api/routes/projects_route.py` with homography endpoints:

### Endpoints

**POST `/projects/{project_id}/homography/session`**

- Create or get active session for project
- Response: `HomographySessionPublic`

**GET `/projects/{project_id}/homography/session`**

- Get current homography session
- Returns 404 if none exists
- Response: `HomographySessionPublic`

**POST `/homography/sessions/{session_id}/pairs`**

- Add point pair to session
- Body: `HomographyPairCreate`
- Response: `HomographyPairPublic`

**PUT `/homography/sessions/{session_id}/pairs`**

- Replace all pairs (bulk update)
- Body: `List[HomographyPairCreate]`
- Response: `List[HomographyPairPublic]`

**DELETE `/homography/pairs/{pair_id}`**

- Delete specific pair
- Response: 204

**POST `/homography/sessions/{session_id}/solve`**

- Validate ≥4 pairs exist
- Call `solve_homography_from_pairs()`
- Update session status to 'solved'
- Store matrix in homography_model table
- Response: `HomographySolveResponse` with matrix

**GET `/homography/sessions/{session_id}/model`**

- Get solved homography model
- Response: `HomographyModelPublic`

**GET `/homography/sessions/{session_id}/export`**

- Export in process-video compatible JSON format
- Format matches `homography-points.json` structure
- Response: JSON matching `{pairs: [...], imagesMeta: {...}, mapMeta: {...}}`

**POST `/projects/{project_id}/extract-frame`**

- Extract first frame from project video using OpenCV
- Create `media_asset` with `kind='image'`
- Link to homography session
- Response: `MediaAssetPublic`

## Frontend: API Client Regeneration

Run `./scripts/generate-client.sh` to update TypeScript SDK in `frontend/app/client/`.

## Frontend: Integration with HomographyPicker

Update `frontend/app/homography/HomographyPicker.tsx`:

### Changes needed:

1. **Accept props**: `projectId: string`, `existingSession?: HomographySessionPublic`

2. **Load session on mount**:

   - Fetch existing session via `GET /projects/{projectId}/homography/session`
   - Load pairs into state
   - Load screenshot asset if available

3. **Screenshot handling**:

   - Add button "Extract Video Frame" → calls `POST /projects/{projectId}/extract-frame`
   - Display extracted frame instead of file upload dropzone when available
   - Allow user to manually upload frame if extraction fails

4. **Save pairs to backend**:

   - On pair creation/deletion, debounce and call `PUT /homography/sessions/{session_id}/pairs`
   - Show save status indicator (saving, saved, error)

5. **Solve button**:

   - Replace "Download JSON" with "Solve Homography" button
   - Enabled when ≥4 pairs exist
   - Calls `POST /homography/sessions/{session_id}/solve`
   - Show loading state during solve
   - Display matrix preview on success (formatted 3x3 grid)
   - Show reprojection error metric

6. **Display solved state**:

   - Badge: "Solved" or "Draft" based on session status
   - If solved, show matrix data in collapsible section
   - Allow editing pairs even after solve (marks as 'draft' again)

### New component: `HomographyMatrixDisplay.tsx`

- Props: `matrix: number[][]`, `error?: number`
- Renders 3x3 matrix in formatted table
- Shows scientific notation for readability
- Displays reprojection error badge

## Frontend: React Query Hooks

Create `frontend/app/hooks/useHomography.ts`:

```typescript
- useHomographySession(projectId) - fetch session
- useCreateSession(projectId) - create new session
- useUpdatePairs(sessionId) - bulk update pairs
- useSolveHomography(sessionId) - trigger solve
- useExtractFrame(projectId) - extract video frame
- useExportHomography(sessionId) - get process-video compatible JSON
```

## Frontend: Integration in Project Detail Page

Update `frontend/app/routes/projects.$projectId.tsx`:

1. Add "Homography" tab/section
2. Render `<HomographyPicker projectId={projectId} />`
3. Show status badge: "Not configured" | "Draft" | "Solved"
4. Guard video processing behind homography solved status (optional)

## Verification Steps

1. **Backend DB**:

   - Run migration, verify tables created
   - Test CRUD operations via Python shell

2. **Backend API**:

   - Create session for project
   - Add 4+ point pairs
   - Solve homography
   - Verify matrix stored in DB
   - Test export endpoint returns process-video compatible JSON

3. **Frontend Flow**:

   - Open project detail page
   - Click "Extract Frame" or upload screenshot
   - Enter picking mode
   - Create 4+ point pairs
   - Pairs auto-save to backend
   - Click "Solve Homography"
   - Matrix displays with reprojection error
   - Refresh page - pairs and matrix persist

4. **Error Handling**:

   - Try solving with less than 4 pairs (should show error)
   - Test with collinear points (OpenCV should fail gracefully)
   - Verify error messages displayed in UI

5. **Integration Test**:

   - Export homography session as JSON
   - Use exported JSON with `process-video/main.py --homography` flag
   - Verify speed calculation works

## Key Files Modified/Created

**Backend**:

- `backend/src/common/database/models/homography_session_table.py` (new)
- `backend/src/common/database/models/homography_pair_table.py` (new)
- `backend/src/common/database/models/homography_model_table.py` (new)
- `backend/src/common/features/homography/schemas.py` (new)
- `backend/src/common/features/homography/crud.py` (new)
- `backend/src/common/features/homography/solver.py` (new - wraps estimate_distance.py)
- `backend/src/api/routes/projects_route.py` (extend with homography endpoints)

**Frontend**:

- `frontend/app/homography/HomographyPicker.tsx` (modify)
- `frontend/app/homography/HomographyMatrixDisplay.tsx` (new)
- `frontend/app/hooks/useHomography.ts` (new)
- `frontend/app/routes/projects.$projectId.tsx` (modify)

## Dependencies

- Backend: `opencv-python` already exists in process-video, ensure it's available to main backend
- Frontend: No new dependencies needed

### To-dos

- [ ] Create homography database models (session, pair, model tables) and generate Alembic migration
- [ ] Implement OpenCV homography solver with RANSAC and error calculation
- [ ] Create Pydantic schemas and CRUD functions for homography operations
- [ ] Implement REST API endpoints for session management, pairs, and solving
- [ ] Add endpoint to extract first frame from video and save as media asset
- [ ] Regenerate TypeScript client SDK for new homography endpoints
- [ ] Create React Query hooks for homography operations
- [ ] Create HomographyMatrixDisplay component for showing solved matrix
- [ ] Update HomographyPicker to load/save from backend and trigger solve
- [ ] Integrate homography UI into project detail page
- [ ] End-to-end testing: create session, add pairs, solve, verify persistence
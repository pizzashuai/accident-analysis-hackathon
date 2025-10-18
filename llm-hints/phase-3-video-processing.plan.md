<!-- f0f8fe1e-a85c-4a1a-81e1-2e23d67bcae3 610de3f3-2c44-44a2-b5ce-dec626ff8ab2 -->

# Phase 3 - Video Processing Run Integration

## Overview

Enable users to trigger YOLO/ByteTrack video processing via Celery tasks, store all detections in the database, provide real-time WebSocket status updates, and support both live annotation overlay in the React video player and downloadable pre-rendered annotated videos with speed calculations.

## Key Requirements

- **Manual trigger**: Users click "Run Analysis" button (requires solved homography)
- **Detection storage**: All detections stored in `detection` table (videos restricted to <10 seconds)
- **Dual annotation**: Live overlay in React player + button to generate/download annotated video
- **Speed calculation**: Always enabled (homography must be solved before processing)
- **Real-time updates**: WebSocket notifications for processing stage updates

## Backend: Database Models & Migrations

### 1. Create Processing Run Models

**File**: `backend/src/common/database/models/processing_run_table.py`

```python
from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from src.common.database.models.user_table import Base

class ProcessingRun(Base):
    __tablename__ = "processing_run"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    homography_session_id = Column(UUID(as_uuid=True), ForeignKey("homography_session.id"))
    params = Column(JSONB, nullable=False, default={})
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    progress = Column(JSONB, nullable=False, default={})  # {stage: str, percent: int, message: str}
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime)
    error_message = Column(String)

    # Relationships
    project = relationship("Project", back_populates="processing_runs")
    detections = relationship("Detection", back_populates="run", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")
```

### 2. Create Detection Model

**File**: `backend/src/common/database/models/detection_table.py`

```python
from sqlalchemy import Column, Integer, Float, ForeignKey, String, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from src.common.database.models.user_table import Base

class Detection(Base):
    __tablename__ = "detection"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(UUID(as_uuid=True), ForeignKey("processing_run.id", ondelete="SET NULL"))
    frame_idx = Column(Integer, nullable=False)
    t_ms = Column(Integer, nullable=False)
    track_id = Column(Integer)
    cls = Column(String, nullable=False)
    conf = Column(Float)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    w = Column(Float, nullable=False)
    h = Column(Float, nullable=False)
    wx = Column(Float)  # World x (homography-transformed)
    wy = Column(Float)  # World y (homography-transformed)
    extra = Column(JSONB, nullable=False, default={})  # speed_mph, geo coords, etc.

    # Relationships
    run = relationship("ProcessingRun", back_populates="detections")

    __table_args__ = (
        Index("detection_project_time_idx", "project_id", "t_ms"),
        Index("detection_project_track_idx", "project_id", "track_id"),
        Index("detection_run_idx", "run_id"),
        Index("detection_frame_idx", "frame_idx"),
    )
```

### 3. Create Artifact Model

**File**: `backend/src/common/database/models/artifact_table.py`

```python
from sqlalchemy import Column, DateTime, ForeignKey, String, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from src.common.database.models.user_table import Base

class Artifact(Base):
    __tablename__ = "artifact"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(UUID(as_uuid=True), ForeignKey("processing_run.id", ondelete="SET NULL"))
    kind = Column(String, nullable=False)
    uri = Column(String, nullable=False)
    meta = Column(JSONB, nullable=False, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    run = relationship("ProcessingRun", back_populates="artifacts")

    __table_args__ = (
        CheckConstraint(
            "kind IN ('jsonl_detections', 'csv_detections', 'annotated_video', 'report', 'debug')",
            name="artifact_kind_check"
        ),
    )
```

### 4. Update Project Model

**File**: `backend/src/common/database/models/project_table.py` (add relationship)

```python
processing_runs = relationship("ProcessingRun", back_populates="project", cascade="all, delete-orphan")
```

### 5. Update Models Index

**File**: `backend/src/common/database/models/__init__.py`

Add imports for new models.

### 6. Generate Migration

```bash
cd backend
alembic revision --autogenerate -m "add_processing_run_detection_artifact_tables"
alembic upgrade head
```

## Backend: Pydantic Schemas & CRUD

### File: `backend/src/common/features/processing/schemas.py`

```python
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, UUID4

# Processing Run schemas
class ProcessingRunCreate(BaseModel):
    params: Optional[Dict[str, Any]] = {}

class ProcessingRunPublic(BaseModel):
    id: UUID4
    project_id: UUID4
    homography_session_id: Optional[UUID4]
    params: Dict[str, Any]
    status: str
    progress: Dict[str, Any]
    started_at: datetime
    finished_at: Optional[datetime]
    error_message: Optional[str]

class ProcessingRunsPublic(BaseModel):
    data: List[ProcessingRunPublic]
    count: int

# Detection schemas
class DetectionPublic(BaseModel):
    id: int
    frame_idx: int
    t_ms: int
    track_id: Optional[int]
    cls: str
    conf: Optional[float]
    x: float
    y: float
    w: float
    h: float
    wx: Optional[float]
    wy: Optional[float]
    extra: Dict[str, Any]

class DetectionsPublic(BaseModel):
    data: List[DetectionPublic]
    count: int

# Artifact schemas
class ArtifactPublic(BaseModel):
    id: UUID4
    kind: str
    uri: str
    meta: Dict[str, Any]
    created_at: datetime

class ArtifactsPublic(BaseModel):
    data: List[ArtifactPublic]
    count: int

# Progress update schema
class ProcessingProgress(BaseModel):
    stage: str
    percent: int
    message: str
```

### File: `backend/src/common/features/processing/crud.py`

Key functions:

- `create_processing_run(db, project_id, homography_session_id, params) → ProcessingRun`
- `get_processing_run(db, run_id) → ProcessingRun | None`
- `list_processing_runs(db, project_id) → List[ProcessingRun]`
- `update_run_status(db, run_id, status, progress, error_message)`
- `update_run_progress(db, run_id, stage, percent, message)`
- `bulk_insert_detections(db, run_id, detections_list)` - efficient batch insert
- `get_detections_by_run(db, run_id, skip, limit) → List[Detection]`
- `get_detections_by_frame(db, run_id, frame_idx) → List[Detection]`
- `create_artifact(db, project_id, run_id, kind, uri, meta) → Artifact`
- `list_artifacts(db, run_id) → List[Artifact]`

## Backend: Celery Task - Video Processing

### File: `backend/src/worker/celery_app/tasks.py` (add new task)

```python
@celery_app.task(bind=True)
def process_video_task(self, project_id: str, run_id: str):
    """
    Process video with YOLO detection, ByteTrack tracking, and speed calculation.

    Stages:
    1. Download video from S3
    2. Load homography data
    3. Run YOLO + ByteTrack (with progress updates)
    4. Calculate speeds using homography
    5. Bulk insert detections to DB
    6. Upload JSONL artifact to S3
    7. Mark run as completed
    """
    # Implementation wraps backend/src/common/features/process-video/main.py
    # Key steps:
    # - Update progress via update_run_progress() after each stage
    # - Use VideoAnnotator with homography for speed calculation
    # - Store all detections in DB via bulk_insert_detections()
    # - Create JSONL artifact for backup/export
    # - Handle errors and update run status accordingly
```

### File: `backend/src/worker/celery_app/tasks.py` (add annotated video generation task)

```python
@celery_app.task(bind=True)
def generate_annotated_video_task(self, project_id: str, run_id: str):
    """
    Generate pre-rendered annotated video with bounding boxes and speed labels.

    Steps:
    1. Download original video from S3
    2. Load detections from database
    3. Load homography data
    4. Use VideoAnnotator to render annotated video
    5. Upload annotated video to S3
    6. Create artifact record
    """
```

## Backend: WebSocket Support

### File: `backend/src/api/websocket.py` (new)

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, project_id: str):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = set()
        self.active_connections[project_id].add(websocket)

    def disconnect(self, websocket: WebSocket, project_id: str):
        if project_id in self.active_connections:
            self.active_connections[project_id].discard(websocket)

    async def broadcast_to_project(self, project_id: str, message: dict):
        if project_id in self.active_connections:
            for connection in self.active_connections[project_id]:
                await connection.send_json(message)

manager = ConnectionManager()
```

### Update `backend/src/api/main.py`

Add WebSocket endpoint:

```python
@app.websocket("/ws/projects/{project_id}/processing")
async def websocket_processing_updates(websocket: WebSocket, project_id: str):
    await manager.connect(websocket, project_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, project_id)
```

Update Celery task to broadcast progress via WebSocket manager.

## Backend: API Routes

### Extend `backend/src/api/routes/projects_route.py`

**POST `/projects/{project_id}/processing/start`**

- Validate homography is solved (status='solved')
- Validate video exists and duration < 10 seconds
- Create ProcessingRun record with status='pending'
- Enqueue `process_video_task`
- Return ProcessingRunPublic

**GET `/projects/{project_id}/processing/runs`**

- List all processing runs for project
- Return ProcessingRunsPublic

**GET `/processing/runs/{run_id}`**

- Get single run with status and progress
- Return ProcessingRunPublic

**GET `/processing/runs/{run_id}/detections`**

- Paginated detections for a run
- Query params: `skip`, `limit`, `frame_idx` (optional filter)
- Return DetectionsPublic

**GET `/processing/runs/{run_id}/detections/frames/{frame_idx}`**

- Get detections for specific frame
- Return DetectionsPublic

**POST `/processing/runs/{run_id}/generate-video`**

- Validate run is completed
- Enqueue `generate_annotated_video_task`
- Return artifact record

**GET `/processing/runs/{run_id}/artifacts`**

- List artifacts (JSONL, annotated videos, etc.)
- Return ArtifactsPublic

**GET `/artifacts/{artifact_id}/download`**

- Generate presigned S3 URL for artifact download
- Return `{"url": "..."}`

## Frontend: API Client Regeneration

Run `./scripts/generate-client.sh` to update TypeScript SDK.

## Frontend: React Query Hooks

### File: `frontend/app/hooks/useProcessing.ts`

```typescript
- useStartProcessing(projectId) - Start video processing
- useProcessingRuns(projectId) - List runs for project
- useProcessingRun(runId) - Get single run details
- useDetections(runId, frameIdx?) - Fetch detections with optional frame filter
- useGenerateAnnotatedVideo(runId) - Trigger video generation
- useArtifacts(runId) - List artifacts
- useArtifactDownloadUrl(artifactId) - Get download URL
- useProcessingWebSocket(projectId) - WebSocket hook for real-time updates
```

## Frontend: Video Annotation Viewer Component

### File: `frontend/app/components/VideoAnnotation/VideoAnnotationViewer.tsx`

**Features**:

- HTML5 video player with custom controls
- Load detections for current frame from backend
- Render bounding boxes as SVG overlay
- Display track ID and speed labels
- Seek to specific frames
- Toggle annotation visibility
- Handle video player events (play, pause, seek, timeupdate)

**Key implementation**:

```typescript
const VideoAnnotationViewer = ({
  videoUrl,
  runId,
}: {
  videoUrl: string;
  runId: string;
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [showAnnotations, setShowAnnotations] = useState(true);

  const { data: detections } = useDetections(runId, currentFrame);

  // Update frame on timeupdate event
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      const fps = 30; // Get from video metadata
      const frame = Math.floor(video.currentTime * fps);
      setCurrentFrame(frame);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    return () => video.removeEventListener('timeupdate', handleTimeUpdate);
  }, []);

  return (
    <div style={{ position: 'relative' }}>
      <video ref={videoRef} src={videoUrl} controls />
      {showAnnotations && detections && (
        <BoundingBoxOverlay
          detections={detections}
          videoWidth={videoRef.current?.videoWidth}
          videoHeight={videoRef.current?.videoHeight}
        />
      )}
    </div>
  );
};
```

### File: `frontend/app/components/VideoAnnotation/BoundingBoxOverlay.tsx`

Renders SVG overlay with bounding boxes and labels positioned absolutely over video.

## Frontend: Processing Panel Component

### File: `frontend/app/components/Processing/ProcessingPanel.tsx`

**Features**:

- "Run Analysis" button (disabled if homography not solved)
- Display processing runs table with status badges
- Real-time progress bar during processing
- Show stage updates (Downloading, Detecting, Tracking, Calculating speeds, Saving)
- List artifacts with download buttons
- "Generate Annotated Video" button for completed runs
- Error messages display

**WebSocket integration**:

```typescript
const { progress, status } = useProcessingWebSocket(projectId);

useEffect(() => {
  if (progress) {
    setProgressPercent(progress.percent);
    setProgressMessage(progress.message);
  }
}, [progress]);
```

## Frontend: Integration in Project Detail Page

Update `frontend/app/routes/projects.$projectId.tsx`:

1. Add "Processing" tab after "Homography"
2. Render `<ProcessingPanel projectId={projectId} />`
3. Show badge on Overview tab indicating processing status
4. Integrate `VideoAnnotationViewer` in Video tab (only if run completed)

## Verification Steps

### Backend Testing

1. Create project, upload video (<10s), solve homography
2. POST `/projects/{id}/processing/start` → verify run created, Celery task enqueued
3. Monitor processing run status changes: pending → running → completed
4. Check detections table populated with all frames
5. Verify JSONL artifact created in S3
6. Verify speeds calculated in detection.extra
7. Test generating annotated video artifact

### Frontend Testing

1. Open project with solved homography
2. Navigate to Processing tab
3. Click "Run Analysis" → see real-time progress updates via WebSocket
4. Progress bar shows: 0% → 25% (downloading) → 50% (detecting) → 75% (tracking) → 100% (complete)
5. After completion, view detections overlayed on video player
6. Seek through video → bounding boxes update per frame
7. Click "Generate Annotated Video" → download button appears
8. Download and verify annotated video plays with burned-in annotations

### Error Testing

1. Try starting processing without solved homography → error message
2. Try processing video >10 seconds → validation error
3. Simulate processing failure → run marked as failed, error displayed
4. Test WebSocket reconnection on network interruption

## Key Files Summary

**Backend**:

- `backend/src/common/database/models/processing_run_table.py` (new)
- `backend/src/common/database/models/detection_table.py` (new)
- `backend/src/common/database/models/artifact_table.py` (new)
- `backend/src/common/features/processing/schemas.py` (new)
- `backend/src/common/features/processing/crud.py` (new)
- `backend/src/worker/celery_app/tasks.py` (extend)
- `backend/src/api/websocket.py` (new)
- `backend/src/api/main.py` (extend with WebSocket endpoint)
- `backend/src/api/routes/projects_route.py` (extend with processing endpoints)

**Frontend**:

- `frontend/app/hooks/useProcessing.ts` (new)
- `frontend/app/components/VideoAnnotation/VideoAnnotationViewer.tsx` (new)
- `frontend/app/components/VideoAnnotation/BoundingBoxOverlay.tsx` (new)
- `frontend/app/components/Processing/ProcessingPanel.tsx` (new)
- `frontend/app/routes/projects.$projectId.tsx` (extend with Processing tab)

## Dependencies

**Backend**:

- `websockets` (for WebSocket support)
- Existing: `supervision`, `ultralytics`, `opencv-python`, `celery`, `redis`

**Frontend**:

- No new dependencies (use native WebSocket API)

## Implementation Notes

1. **Bulk Insert Performance**: Use SQLAlchemy `bulk_insert_mappings()` for efficient detection insertion
2. **WebSocket Connection**: Establish connection when user opens Processing tab, close on tab unmount
3. **Video Duration Check**: Use `cv2.VideoCapture` to validate <10 seconds before processing
4. **Homography Validation**: Query `homography_session.status='solved'` before allowing processing start
5. **Speed Calculation**: Reuse `DistanceEstimator` from `process-video/src/estimate_distance.py`
6. **JSONL Backup**: Store JSONL artifact for potential re-import or debugging
7. **Presigned URLs**: Use same S3 presigned URL generation for artifact downloads

### To-dos

- [x] Create ProcessingRun, Detection, and Artifact SQLAlchemy models with proper relationships and indexes
- [x] Generate and verify Alembic migration for processing_run, detection, and artifact tables
- [x] Create Pydantic schemas and CRUD functions in features/processing/ with bulk insert support
- [x] Implement process_video_task wrapping process-video pipeline with progress updates and detection storage
- [x] Implement generate_annotated_video_task for creating downloadable annotated videos
- [x] Add WebSocket support with ConnectionManager for real-time processing updates
- [x] Implement processing API endpoints (start, status, detections, artifacts, generate video)
- [x] Regenerate TypeScript client SDK for new processing endpoints
- [x] Create React Query hooks for processing operations and WebSocket connection
- [x] Build VideoAnnotationViewer component with live detection overlay and frame synchronization
- [x] Create ProcessingPanel component with run status, progress bar, and WebSocket integration
- [x] Integrate Processing tab into project detail page with status badges
- [ ] End-to-end testing: trigger processing, verify real-time updates, test live overlay and video download

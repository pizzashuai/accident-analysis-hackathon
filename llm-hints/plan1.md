I'm working on a accident analysis project and designing the database schema for postgres. I have the user table, and no multi-org setup, one user owns projects. Each project is an accident analysis, has title, description... On a high level, user logs in, creates/crud project, on each project, user uploads a cctv accident video, specifies location (address) or pin on the google map, and our backend takes a screenshot of the accident video, and user selects a set of points pairs (video screenshot to google map 2d) so backend performs homography. Eventually user will get a list of events that backend infers from reading the video after detect/track vehicles, vehicle gets a tracking id. The events are sortable and has description text and timestamp. There will be jsonl for all the detected objects/vehicles for each frame (now I'm storing as jsonl in local test, but I can also store differently in the database), user needs to be able to select the relevant tracking id, so only relevant bbox are annotated in the video. Help me design high level tables in postgres w/ minimal code

./backend/src/common/features/process-video has all the code that can run as script to process the video, it's saving everything as file and not designed to run as interfaces with database yet.
.../process-video/happy1.mp4 is a sample video to analyze, and we should use its first frame as the screenshot to act as the image to pick points for homography.
./backend/src/common/database/models contains all the postgres table schema definition

• Phase 1 — Projects & Media Backbone

- Goal: let a logged-in user create/manage accident projects, upload the raw CCTV video, and capture the scene location.
- Backend: add SQLAlchemy models/migrations for project, media_asset, project_location inside backend/src/common/database/models; build CRUD & upload endpoints
  under backend/src/api/routes/projects_route.py storing uploads to a local bucket (or temp folder) and linking to media_asset; ensure Alembic migration +
  dependency wiring.
- Frontend: add projects dashboard & create/edit flow in frontend/app/routes/home.tsx (or nested routes), reuse Mantine forms for metadata, use XHR upload
  component to hit the new API, surface project cards with location badges.
- Glue: after backend changes, run ./scripts/generate-client.sh so frontend/app/client/sdk.gen.ts exposes the new endpoints; update React Query hooks.
- Verify (frontend): log in → create project with title/description, set address/pin (leaflet/geocode stub), upload sample video, see project listed with
  stored details; refresh to confirm persistence.

  Phase 1.5 — S3 Media Pipeline & Frame Snapshot

  - Goal: move video storage to S3, stream/play it in the UI, and capture the first frame automatically for downstream homography work.
  - Backend: load AWS credentials (access key & secret) from .env, initialize a boto3 client, issue presigned POST/GET URLs, and update upload endpoint to
    orchestrate direct S3 uploads while persisting asset metadata (duration, fps) gathered via a probe. After upload confirmation, trigger a task that downloads
    the video via signed URL, grabs frame 0 with OpenCV, pushes the PNG back to S3, and saves a media_asset row (kind='image') marked as the project’s default
    screenshot.
  - Frontend: swap the upload control to a presigned S3 flow (get fields → POST to S3 → notify backend), embed the stored S3 URL (via signed GET/CloudFront) in a
    <video> player on the project detail page, and show the auto-generated screenshot once the background job finishes, with progress feedback.
  - Glue: manage S3 region/bucket config via settings + .env, update docker-compose for optional local MinIO, regenerate TS client.
  - Verify (frontend): upload video → UI shows “processing frame” → after snapshot task completes, player streams from S3 and screenshot thumbnail appears;
    refresh confirms assets persist.

Phase 2 — Homography Capture & Solving

- Goal: enable storing the screenshot + point pairs from the existing Homography Picker and solve the homography matrix server-side.
- Backend: introduce homography_session, homography_pair, homography_model tables plus API in projects_route or new homography_route; add endpoint to request
  “extract first frame” that reuses backend/src/common/features/process-video/main.py utilities to grab a frame, save as media_asset; implement solve endpoint
  that runs OpenCV homography fit and stores the 3x3 in homography_model.
- Frontend: adapt frontend/app/homography/HomographyPicker.tsx to load/save sessions via the generated client, allow selecting a stored screenshot/map
  snapshot, persist pairs, trigger solve, and show matrix/result status inside the existing UI.
- Glue: regenerate client; add toast/error handling; guard routes so each session tied to selected project.
- Verify (frontend): open project detail → click “Set Homography” → fetches stored frame, pick ≥4 correspondences, save, run solve; page shows “Solved” with
  matrix preview, reloading preserves pairs.

Phase 3 — Video Processing Run Integration

- Goal: run the YOLO/ByteTrack pipeline through Celery, persist detections, and expose live annotation data (JSONL-style) for the frontend overlay.
- Backend: add processing_run, detection, artifact tables and CRUD; wrap process_video_with_supervision from backend/src/common/features/process-video/main.py
  into a Celery task under backend/src/worker; task should read the project’s video asset, reuse stored homography (optional), write JSONL outputs, bulk-insert
  detections, and create artifact rows; expose REST endpoints to start a run, poll status, stream/paginate detection frames, and fetch aggregate stats (e.g.,
  track counts).
- Frontend: create a Processing panel on project detail to start a run (maybe configurable thresholds), show status (pending/running/completed/failed) with
  polling, list key metrics, surface artifact downloads, and hydrate the new annotation component with the JSONL-esque detection feed.
- Glue: ensure Celery worker uses same settings (update docker-compose if needed); run ./scripts/generate-client.sh; keep JSONL schema aligned with
  backend/src/common/features/process-video utilities so frontend and backend share the same structures.
- Verify (frontend): project detail → click “Run Analysis” → status transitions to running then completed; detection summary/track counts appear; video overlay
  renders live boxes sourced from the run without needing a pre-rendered annotated video.

Phase 4 — Event Review & Track Selection

- Goal: let users curate relevant tracks, update annotation overlays interactively, and review inferred events per project.
- Backend: add selected_track and event endpoints returning paginated detections/events filtered by project or track; implement track-selection API to toggle
  selected_track and immediately influence detection feeds; extend processing task to calculate basic events (e.g., stop, collision) from JSONL outputs or a
  simplified heuristic (store in event).
- Frontend: add tabs for “Detections”, “Tracks”, “Events”; embed the video annotation component so clicking tracks (list/table) instantly toggles bounding
  boxes on the playing video without producing a new video file; render detection timeline (frame/time, class, confidence) and event list filtered by selected
  tracks/timestamps.
- Glue: regenerate client; add React Query caches plus websocket/polling updates for track selections; ensure throughput manageable (lazy load, server-side
  pagination).
- Verify (frontend): after processing run, open Events tab → select/deselect track IDs and see overlay update immediately while persisting selection; filter
  events by chosen tracks and refresh to confirm state sync.

Phase 5 — Reporting & UX Polish

- Goal: ship a cohesive reviewer experience with exports and map context.
- Backend: extend artifact handling for CSV/JSON exports; expose endpoint for summarized metrics (speed, Δv) derived from process-video outputs; enforce
  permissions and validation across routes.
- Frontend: add download buttons, summary cards (total events, impact speed), embed map view with selected tracks overlay; enhance error/loading states and
  guard rails.
- Verify (frontend): user can walk entire flow—create project → set location → capture homography → run processing → review events/metrics → download reports—
  without console/network errors.

Next steps: once ready to start, tackle Phase 1 migration scaffolding first, keeping migrations/tests in sync.

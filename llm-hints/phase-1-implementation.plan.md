<!-- f9845890-5a5e-4d07-b143-8c187dbe2cbc 33d7fad6-88a9-4d6b-86f3-c7177ecaf414 -->
# Phase 1 - Projects & Media Backbone Implementation

## Backend: Database Models & Migrations

**Create new SQLAlchemy models** in `backend/src/common/database/models/`:

- `project_table.py` - Project model with `id`, `user_id`, `title`, `description`, `created_at`, optional `video_id` FK
- `media_asset_table.py` - MediaAsset model with `id`, `project_id`, `kind` (enum: video/image/map_snapshot/json), `uri`, `bytes`, `meta` (JSONB), `created_at`
- `project_location_table.py` - ProjectLocation model with `project_id` (PK), `addr_line`, `lat`, `lon`, `source`

**Update** `backend/src/common/database/models/__init__.py` to export new models for Alembic discovery.

**Generate Alembic migration** using `alembic revision --autogenerate -m "add_project_media_location_tables"` to create the three tables with proper relationships and indexes.

## Backend: Pydantic Schemas & CRUD

**Create** `backend/src/common/features/project/` directory with:

- `schemas.py` - Pydantic models: `ProjectCreate`, `ProjectUpdate`, `ProjectPublic`, `ProjectsPublic`, `MediaAssetPublic`, `ProjectLocationCreate`, `ProjectLocationPublic`
- `crud.py` - CRUD functions: `create_project()`, `get_project()`, `list_projects()`, `update_project()`, `delete_project()`, `create_media_asset()`, `upsert_project_location()`
- `__init__.py` - Export schemas for clean imports

## Backend: API Routes

**Create** `backend/src/api/routes/projects_route.py` with endpoints:

- `POST /projects` - Create project (requires auth, links to current user)
- `GET /projects` - List user's projects with pagination
- `GET /projects/{project_id}` - Get single project with media assets & location
- `PATCH /projects/{project_id}` - Update project metadata
- `DELETE /projects/{project_id}` - Delete project (cascade deletes media/location)
- `POST /projects/{project_id}/upload-video` - Handle multipart file upload, save to `backend/uploads/{project_id}/`, create MediaAsset record, link as project.video_id
- `POST /projects/{project_id}/location` - Create/update project location

**Update** `backend/src/api/main.py` to include the new projects router.

**Create uploads directory structure** - Ensure `backend/uploads/` exists for local file storage.

## Frontend: API Client Regeneration

**Run** `./scripts/generate-client.sh` to regenerate TypeScript client SDK with new project endpoints in `frontend/app/client/`.

## Frontend: Projects Dashboard

**Create** `frontend/app/routes/projects.tsx` - Main projects dashboard:

- Display project cards in a grid using Mantine `Card` and `Grid`
- Show title, description, created date, location badge if set
- "Create Project" button opening modal
- Click card to navigate to project detail

**Create** `frontend/app/routes/projects.$projectId.tsx` - Project detail page:

- Display project metadata (title, description, location)
- Video player if video uploaded (use HTML5 `<video>` tag)
- "Edit Project" and "Delete Project" buttons
- Location display with address/coordinates
- Upload video section with drag-drop or file picker

**Create** `frontend/app/components/Projects/` directory with:

- `CreateProjectModal.tsx` - Mantine modal with form (title, description, location fields with optional geocoding stub)
- `ProjectCard.tsx` - Reusable project card component
- `VideoUpload.tsx` - File upload component with progress bar using `@mantine/dropzone`
- `LocationPicker.tsx` - Simple form with address input and lat/lon fields (map integration can be basic leaflet or deferred)

**Update** `frontend/app/routes.ts` to add:

```typescript
route('projects', 'routes/projects.tsx'),
route('projects/:projectId', 'routes/projects.$projectId.tsx'),
```

**Update** `frontend/app/components/Common/Navbar.tsx` to add "Projects" navigation link.

## Frontend: React Query Hooks

**Create** `frontend/app/hooks/useProjects.ts` with:

- `useProjects()` - Fetch user's projects list
- `useProject(projectId)` - Fetch single project
- `useCreateProject()` - Mutation for creating project
- `useUpdateProject()` - Mutation for updating project
- `useDeleteProject()` - Mutation for deleting project
- `useUploadVideo()` - Mutation for video upload with progress tracking

## Verification Steps

1. Start backend and frontend services
2. Log in as existing user
3. Navigate to Projects page
4. Create new project with title "Test Accident" and description
5. Set location (address or lat/lon)
6. Upload sample video from `backend/src/common/features/process-video/happy1.mp4`
7. Verify project appears in list with metadata
8. Refresh page - confirm persistence
9. Click project card to view detail page
10. Verify video plays and location displays correctly

### To-dos

- [x] Create SQLAlchemy models for project, media_asset, and project_location tables
- [x] Generate and verify Alembic migration for new tables
- [x] Create Pydantic schemas and CRUD functions in features/project/
- [x] Implement projects_route.py with all CRUD and upload endpoints
- [x] Run generate-client.sh to update TypeScript SDK
- [x] Create reusable project components (ProjectCard, CreateProjectModal, VideoUpload, LocationPicker)
- [x] Create React Query hooks for project operations
- [x] Implement projects list and detail pages with routing
- [x] Update Navbar to include Projects link
- [x] Test complete flow: create project, set location, upload video, verify persistence
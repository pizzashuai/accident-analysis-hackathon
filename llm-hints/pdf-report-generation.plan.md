<!-- 0641251d-8bdd-48e8-86d6-7bea6706f18d 9c32e23e-7851-4c5b-b629-ec06043f1e18 -->
# PDF Report Generation Implementation

## Overview

Add PDF report generation feature that creates professional accident analysis reports with LLM analysis timeline, video screenshots, and map overlays.

## Backend Implementation

### 1. Database Schema

**File**: `backend/src/common/database/models/report_table.py` (new)

- Create `Report` model with fields:
- `id`, `project_id`, `run_id`, `analysis_id`
- `status` (pending/generating/completed/failed)
- `pdf_uri` (S3 location)
- `meta` (JSON with analysis metadata, screenshots, map data)
- `created_at`, `completed_at`
- Relationship to Project and ProcessingRun

**File**: `backend/src/common/database/alembic/versions/xxx_add_report_table.py` (new)

- Create Alembic migration to add report table

**File**: `backend/src/common/database/models/__init__.py`

- Export Report model

### 2. PDF Generation Module

**File**: `backend/src/common/features/report/pdf_generator.py` (new)

- Install dependencies: `weasyprint` or `reportlab` + `pillow`
- Create `generate_pdf_report()` function:
- Accept LLM analysis text, screenshots (video + map), metadata
- Use HTML templates with CSS for beautiful formatting
- Include sections: Executive Summary, Timeline, Screenshots, Metadata
- Return PDF bytes

**File**: `backend/src/common/features/report/templates/report_template.html` (new)

- Professional HTML template with CSS styling
- Sections: Header with project info, Analysis Summary, Timeline, Screenshots side-by-side, Footer

**File**: `backend/src/common/features/report/screenshot_generator.py` (new)

- Extract collision detection logic from `backend/src/common/features/screenshot/`
- Create `generate_collision_screenshots()`:
- Find collision frame from LLM analysis timeline
- Generate video screenshot at collision timestamp
- Generate map overlay with trajectories using Google Maps API
- Return both image paths/bytes

### 3. CRUD Operations

**File**: `backend/src/common/features/report/crud.py` (new)

- `create_report()` - Create report record
- `get_report()` - Get report by ID
- `list_reports_by_project()` - List all reports for a project
- `update_report_status()` - Update generation status
- `update_report_pdf_uri()` - Update PDF S3 URI after upload

### 4. Celery Task

**File**: `backend/src/worker/celery_app/tasks.py`

- Add `generate_pdf_report_task()`:

1. Retrieve LLM analysis from Redis/Database
2. Download video from S3
3. Generate collision screenshots (video + map)
4. Generate PDF using pdf_generator
5. Upload PDF to S3
6. Update report record with PDF URI and status
7. Publish completion event (optional)

### 5. API Routes

**File**: `backend/src/api/routes/reports_route.py` (new)

- `POST /api/v1/projects/{project_id}/reports/generate`
- Body: `{analysis_id: str, run_id: str}`
- Validates project ownership
- Creates report record
- Triggers Celery task
- Returns report_id

- `GET /api/v1/projects/{project_id}/reports`
- List all reports for project
- Returns list with status, created_at, pdf_url (presigned)

- `GET /api/v1/projects/{project_id}/reports/{report_id}`
- Get specific report details
- Returns report with presigned PDF URL

- `GET /api/v1/projects/{project_id}/reports/{report_id}/download`
- Download PDF directly (redirect to presigned URL)

**File**: `backend/src/api/main.py`

- Register reports router

## Frontend Implementation

### 1. API Client

**File**: Regenerate OpenAPI client

- Run `pnpm run generate-client` after backend API changes

### 2. React Hook

**File**: `frontend/app/hooks/useReports.ts` (new)

- `useGenerateReport(projectId)` - Mutation to generate report
- `useReports(projectId)` - Query to list reports
- `useReport(projectId, reportId)` - Query to get report details

### 3. UI Component

**File**: `frontend/app/components/VideoAnnotation/ReportGenerationPanel.tsx` (new)

- Button: "Generate PDF Report"
- Shows loading state during generation
- Lists previously generated reports with:
- Timestamp
- Status badge
- Download button
- Uses Mantine Card, Button, Badge, Table components

**File**: `frontend/app/routes/projects.$projectId.tsx`

- Add ReportGenerationPanel to the project detail page
- Display below LLM Analysis Panel
- Only show if LLM analysis is completed

### 4. Polish

- Add error handling and toast notifications
- Add loading states and progress indicators
- Responsive design for mobile/desktop

## Dependencies

**Backend** (`backend/pyproject.toml`):

- `weasyprint>=61.0` (or `reportlab>=4.0`)
- `pillow>=10.0` (image handling)

**Frontend**: No new dependencies needed (using existing Mantine UI)

## Key Files to Reference

- LLM Analysis: `backend/src/common/features/postprocess/llm_agent.py`
- Screenshot Generation: `backend/src/common/features/screenshot/map_overlay.py`
- Celery Tasks: `backend/src/worker/celery_app/tasks.py`
- Existing Routes Pattern: `backend/src/api/routes/llm_analysis_route.py`

### To-dos

- [ ] Create Report database model and Alembic migration
- [ ] Implement PDF generation module with HTML templates
- [ ] Create screenshot generator for collision frames and map overlays
- [ ] Implement CRUD operations for reports
- [ ] Add Celery task for async PDF generation
- [ ] Create API routes for report generation and retrieval
- [ ] Regenerate OpenAPI client with new routes
- [ ] Create React hooks for report operations
- [ ] Build ReportGenerationPanel UI component and integrate into project page
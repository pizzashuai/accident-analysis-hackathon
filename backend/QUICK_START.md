# Quick Start Guide - Unified Backend

## 🚀 Getting Started

### One-Time Setup

```bash
cd backend

# Install all dependencies
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate
```

### Configure Your Editor

Point your IDE/editor to use the unified virtual environment:

- **Python Interpreter**: `backend/.venv/bin/python`
- **PYTHONPATH** is automatically handled by run scripts

## 🏃 Running Services

### Development Mode (Recommended)

**Run both services:**

```bash
./scripts/dev.sh up
```

**Run individually:**

```bash
./scripts/dev.sh api      # API only (foreground)
./scripts/dev.sh worker   # Worker only (foreground)
```

**Stop all:**

```bash
./scripts/dev.sh down
```

**Check status:**

```bash
./scripts/dev.sh logs
```

### Alternative: Individual Scripts

```bash
# API Service
./scripts/run-api.sh

# Worker Service
./scripts/run-worker.sh
```

## 🐳 Docker

### Building Images

Both services use the unified dependency set from root `pyproject.toml`:

```bash
# From repository root
docker build -f backend/services/api/Dockerfile -t my-api:latest backend/
docker build -f backend/services/worker/Dockerfile -t my-worker:latest backend/
```

Each image automatically installs only its required dependencies:

- API image: base + api dependencies
- Worker image: base + worker dependencies

## 📦 Managing Dependencies

### Adding Dependencies

Edit `backend/pyproject.toml`:

**For shared dependencies** (used by both services):

```toml
dependencies = [
    "sqlmodel>=0.0.21,<1",
    "your-new-package>=1.0",  # Add here
]
```

**For API-only dependencies**:

```toml
[project.optional-dependencies]
api = [
    "fastapi[standard]>=0.114,<1",
    "your-api-package>=1.0",  # Add here
]
```

**For worker-only dependencies**:

```toml
[project.optional-dependencies]
worker = [
    "celery>=5.3,<6",
    "your-worker-package>=1.0",  # Add here
]
```

**For dev dependencies**:

```toml
[tool.uv]
dev-dependencies = [
    "pytest<8.0.0,>=7.4.3",
    "your-dev-tool>=1.0",  # Add here
]
```

Then sync:

```bash
uv sync --all-extras
```

## 🧪 Testing

```bash
# Run all tests
./scripts/test.sh

# Run with coverage
uv run pytest --cov
```

## 🔧 Linting & Formatting

```bash
# Format code
./scripts/format.sh

# Run linter
./scripts/lint.sh
```

## 📝 Database Migrations

```bash
# Create a new migration
./scripts/run-alembic.sh revision --autogenerate -m "Your migration message"

# Apply migrations
./scripts/run-alembic.sh upgrade head
```

## 🌐 API Access

When running locally:

- **API Base**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **API OpenAPI**: http://localhost:8000/api/v1/openapi.json

## 🔑 Key Files

| File                          | Purpose                                       |
| ----------------------------- | --------------------------------------------- |
| `pyproject.toml`              | **All dependencies** (single source of truth) |
| `uv.lock`                     | Locked dependency versions                    |
| `libs/backend_db/config.py`   | Shared configuration                          |
| `services/api/core/config.py` | API-specific config (extends base)            |
| `scripts/dev.sh`              | Unified dev helper                            |
| `services/*/Dockerfile`       | Docker build configs                          |

## 🐛 Troubleshooting

### Import errors?

Make sure you're using the root virtual environment:

```bash
cd backend
source .venv/bin/activate
```

### Services won't start?

Check PYTHONPATH is set correctly in the run scripts:

- API: `libs:services/api`
- Worker: `libs:services/worker`

### Docker build fails?

Ensure you're building from the repository root and passing the `backend/` context:

```bash
docker build -f backend/services/api/Dockerfile backend/
```

### Dependencies out of sync?

Re-sync from the root:

```bash
cd backend
uv sync --all-extras
```

## 📚 More Information

See `REFACTORING_SUMMARY.md` for detailed information about the refactoring and architecture.

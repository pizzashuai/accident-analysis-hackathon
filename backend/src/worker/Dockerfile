FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /root/.local/bin/uv --version

# Add uv to PATH
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies from pyproject.toml
RUN /root/.local/bin/uv pip install --system --no-cache -e .

# Copy application source
COPY src /app/src

# Set Python path
ENV PYTHONPATH=/app/src

# Run the worker
CMD ["celery", "-A", "src.worker.celery_app.app", "worker", "-l", "info"]

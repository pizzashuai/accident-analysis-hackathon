#!/usr/bin/env bash
set -e
set -x

cd "$(dirname "$0")/.."
uv run celery -A src.worker.celery_app.app worker -l info

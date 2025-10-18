#!/usr/bin/env bash
set -e
set -x

cd "$(dirname "$0")/.."
uv run uvicorn src.api.main:app --reload

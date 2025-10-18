#! /usr/bin/env bash

set -e
set -x

export PYTHONPATH="$(pwd)"

# Let the DB start
python -m src.api.pre_start.backend_pre_start

# Run migrations
alembic upgrade head

# Create initial data in DB
python -m src.api.pre_start.initial_data

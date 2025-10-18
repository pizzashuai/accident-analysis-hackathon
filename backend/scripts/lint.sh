#!/usr/bin/env bash

set -e
set -x

mypy api database/backend_database
ruff check api database/backend_database
ruff format api database/backend_database --check

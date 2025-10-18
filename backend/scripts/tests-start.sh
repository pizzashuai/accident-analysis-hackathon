#! /usr/bin/env bash
set -e
set -x

export PYTHONPATH="$(pwd)/src"

python -m src.api.pre_start.tests_pre_start

bash scripts/test.sh "$@"

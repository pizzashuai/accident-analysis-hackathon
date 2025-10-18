#!/usr/bin/env bash

set -e
set -x

export PYTHONPATH="$(pwd)/src"

coverage run -m pytest tests/
coverage report
coverage html --title "${@-coverage}"

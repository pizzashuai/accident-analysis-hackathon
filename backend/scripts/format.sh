#!/bin/sh -e
set -x

ruff check api database/backend_database scripts --fix
ruff format api database/backend_database scripts

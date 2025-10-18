#!/usr/bin/env bash
# Unified development script to run both API and Worker services
# Usage:
#   ./scripts/dev.sh up      - Start both API and worker in background
#   ./scripts/dev.sh api     - Start only API service
#   ./scripts/dev.sh worker  - Start only worker service
#   ./scripts/dev.sh down    - Stop all running services
#   ./scripts/dev.sh logs    - Show logs from running services

set -e

cd "$(dirname "$0")/.."

API_PID_FILE=".api.pid"
WORKER_PID_FILE=".worker.pid"

start_api() {
    echo "Starting API service..."
    uv run uvicorn src.api.main:app --reload &
    echo $! > "$API_PID_FILE"
    echo "API service started (PID: $(cat $API_PID_FILE))"
}

start_worker() {
    echo "Starting Worker service..."
    export PYTHONPATH="$(pwd)/src"
    uv run celery -A src.worker.celery_app.app worker -l info &
    echo $! > "$WORKER_PID_FILE"
    echo "Worker service started (PID: $(cat $WORKER_PID_FILE))"
}

stop_services() {
    echo "Stopping services..."
    if [ -f "$API_PID_FILE" ]; then
        kill $(cat "$API_PID_FILE") 2>/dev/null || true
        rm "$API_PID_FILE"
        echo "API service stopped"
    fi
    if [ -f "$WORKER_PID_FILE" ]; then
        kill $(cat "$WORKER_PID_FILE") 2>/dev/null || true
        rm "$WORKER_PID_FILE"
        echo "Worker service stopped"
    fi
}

case "${1:-up}" in
    up)
        echo "Starting all services..."
        start_api
        start_worker
        echo ""
        echo "Services running! Use './scripts/dev.sh down' to stop."
        echo "API: http://localhost:8000"
        echo "API docs: http://localhost:8000/docs"
        ;;
    api)
        start_api
        wait
        ;;
    worker)
        start_worker
        wait
        ;;
    down)
        stop_services
        ;;
    logs)
        echo "Showing process status..."
        if [ -f "$API_PID_FILE" ]; then
            echo "API (PID: $(cat $API_PID_FILE)): Running"
        else
            echo "API: Stopped"
        fi
        if [ -f "$WORKER_PID_FILE" ]; then
            echo "Worker (PID: $(cat $WORKER_PID_FILE)): Running"
        else
            echo "Worker: Stopped"
        fi
        ;;
    *)
        echo "Usage: $0 {up|api|worker|down|logs}"
        echo ""
        echo "Commands:"
        echo "  up      - Start both API and worker in background"
        echo "  api     - Start only API service (foreground)"
        echo "  worker  - Start only worker service (foreground)"
        echo "  down    - Stop all running services"
        echo "  logs    - Show status of running services"
        exit 1
        ;;
esac

import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class HealthHandler(BaseHTTPRequestHandler):
    """Serve simple health responses so App Runner has an HTTP probe target."""

    # Reuse a static JSON body to keep responses lightweight.
    _openapi_stub = json.dumps({"status": "ok"}).encode("utf-8")

    def do_GET(self) -> None:  # noqa: N802 (uppercase by design for http.server)
        path = self.path.rstrip("/") or "/"
        if path in {"", "/", "/health", "/api/v1/openapi.json"}:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self._openapi_stub)))
            self.end_headers()
            self.wfile.write(self._openapi_stub)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A003 (http.server signature)
        # Silence default access logs; Celery logging is noisy enough.
        return


def _start_http_server() -> ThreadingHTTPServer:
    host = os.getenv("WORKER_HEALTH_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_HEALTH_PORT", os.getenv("PORT", "8000")))
    server = ThreadingHTTPServer((host, port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _start_celery_worker() -> subprocess.Popen:
    concurrency = os.getenv("CELERY_CONCURRENCY")
    log_level = os.getenv("CELERY_LOG_LEVEL", "info")

    command = [
        "celery",
        "-A",
        "src.worker.celery_app.app",
        "worker",
        "-l",
        log_level,
    ]
    if concurrency:
        command.extend(["-c", concurrency])

    return subprocess.Popen(command)


def main() -> int:
    http_server = _start_http_server()
    celery_proc = _start_celery_worker()

    def _signal_handler(signum, _frame):
        # Forward termination signals to Celery and stop the HTTP server.
        if celery_proc.poll() is None:
            celery_proc.send_signal(signum)
        http_server.shutdown()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        return_code = celery_proc.wait()
    finally:
        http_server.shutdown()

    return return_code


if __name__ == "__main__":
    sys.exit(main())

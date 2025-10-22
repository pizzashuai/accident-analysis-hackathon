import os
import ssl
import sys
from pathlib import Path

from celery import Celery

# Add project root to Python path to ensure imports work from any working directory
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _redis_ssl_options() -> dict[str, object] | None:
    """Map optional TLS env vars to Celery/Kombu SSL options."""

    use_tls = os.getenv("REDIS_USE_TLS")
    if use_tls is None:
        # Allow configuring TLS solely via cert reqs to support existing setups.
        tls_markers = (
            "REDIS_SSL_CERT_REQS",
            "REDIS_SSL_CA_CERTS",
            "REDIS_SSL_CERTFILE",
            "REDIS_SSL_KEYFILE",
        )
        if not any(os.getenv(env) for env in tls_markers):
            return None
    elif use_tls.lower() not in {"1", "true", "yes", "on"}:
        return None

    cert_reqs_env = os.getenv("REDIS_SSL_CERT_REQS", "CERT_REQUIRED").upper()
    try:
        cert_reqs = getattr(ssl, cert_reqs_env)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported REDIS_SSL_CERT_REQS value: {cert_reqs_env}"
        ) from exc

    ssl_options: dict[str, object] = {"cert_reqs": cert_reqs}
    for key, env_name in (
        ("ca_certs", "REDIS_SSL_CA_CERTS"),
        ("certfile", "REDIS_SSL_CERTFILE"),
        ("keyfile", "REDIS_SSL_KEYFILE"),
    ):
        value = os.getenv(env_name)
        if value:
            ssl_options[key] = value

    return ssl_options


# Use environment variable for Redis URL, default to localhost for local development
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("worker", broker=redis_url, backend=redis_url)

ssl_options = _redis_ssl_options()
if ssl_options:
    # Apply TLS settings to both broker and result backend.
    app.conf.update(
        broker_use_ssl=ssl_options,
        redis_backend_use_ssl=ssl_options,
    )

# Discover tasks within the same package regardless of its import path.
package_root = __name__.rsplit(".", 1)[0]
app.autodiscover_tasks([package_root])

# Explicitly import tasks to ensure they are registered
from . import tasks

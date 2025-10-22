import os
import sys
from pathlib import Path

# Add project root to Python path to ensure imports work from any working directory
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from celery import Celery

from src.common.redis_tls import redis_connection_config


# Use environment variable for Redis URL, default to localhost for local development
raw_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_url, ssl_options = redis_connection_config(raw_redis_url)

app = Celery("worker", broker=redis_url, backend=redis_url)

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

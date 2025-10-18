import os
from celery import Celery

# Use environment variable for Redis URL, default to localhost for local development
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("worker", broker=redis_url)

# Discover tasks within the same package regardless of its import path.
package_root = __name__.rsplit(".", 1)[0]
app.autodiscover_tasks([package_root])

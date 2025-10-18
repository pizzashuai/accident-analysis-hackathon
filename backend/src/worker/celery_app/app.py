import os
import sys
from pathlib import Path
from celery import Celery

# Add project root to Python path to ensure imports work from any working directory
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use environment variable for Redis URL, default to localhost for local development
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("worker", broker=redis_url, backend=redis_url)

# Discover tasks within the same package regardless of its import path.
package_root = __name__.rsplit(".", 1)[0]
app.autodiscover_tasks([package_root])

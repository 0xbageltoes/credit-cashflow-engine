from celery import Celery
from app.core.config import settings

# Configure Redis SSL settings
broker_use_ssl = {
    'ssl_cert_reqs': None,
    'ssl_ca_certs': None,
    'ssl_certfile': None,
    'ssl_keyfile': None
}

redis_backend_use_ssl = broker_use_ssl.copy()

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    broker_connection_retry_on_startup=True
)

# Configure SSL settings
celery_app.conf.update(
    broker_use_ssl=broker_use_ssl,
    redis_backend_use_ssl=redis_backend_use_ssl
)

# Configure task routing
celery_app.conf.task_routes = {
    "app.worker.test_celery": "main-queue"
}

# Configure general settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=1,  # Use single worker
    worker_pool="solo",    # Use solo pool for Windows
    task_track_started=True,  # Track when tasks are started
    task_time_limit=3600,  # 1 hour timeout
    task_soft_time_limit=1800,  # 30 minute soft timeout
    worker_max_tasks_per_child=100  # Restart worker after 100 tasks
)

if __name__ == "__main__":
    celery_app.start()

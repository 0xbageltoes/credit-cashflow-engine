from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    broker_use_ssl={
        'ssl_cert_reqs': 'required',
    },
    redis_backend_use_ssl={
        'ssl_cert_reqs': 'required',
    }
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Optional configuration to handle large results
celery_app.conf.update(
    result_expires=3600,  # Results expire in 1 hour
    task_compression='gzip',  # Compress task messages
    result_compression='gzip',  # Compress results
)

# Configure task routing
celery_app.conf.task_routes = {
    "app.worker.test_celery": "main-queue"
}

# Configure general settings
celery_app.conf.update(
    worker_concurrency=1,  # Use single worker
    worker_pool="solo",    # Use solo pool for Windows
    task_track_started=True,  # Track when tasks are started
    task_time_limit=3600,  # 1 hour timeout
    task_soft_time_limit=1800,  # 30 minute soft timeout
    worker_max_tasks_per_child=100  # Restart worker after 100 tasks
)

if __name__ == "__main__":
    celery_app.start()

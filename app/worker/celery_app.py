"""
Celery Application Configuration

This module configures the Celery application for background task processing.
It includes task routing, periodic task scheduling, and broker configuration.

The configuration is designed for production use with:
- Proper error handling and logging
- Redis as broker and result backend
- Task routing for different types of tasks
- Periodic task scheduling with Celery Beat
"""
import os
from celery import Celery
from celery.schedules import crontab
from celery.signals import setup_logging

from app.core.config import settings

# Configure Celery application
celery_app = Celery(
    "app.worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.worker.monte_carlo_tasks",
        # Add other task modules here
    ]
)

# Optional configuration
celery_app.conf.update(
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Task execution settings
    task_acks_late=True,  # Acknowledge tasks after execution
    worker_prefetch_multiplier=1,  # Process one task at a time
    task_ignore_result=False,  # Store task results
    
    # Task time limits
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3540,  # 59 minutes
    
    # Result backend settings
    result_expires=86400 * 7,  # 7 days
    
    # Task routing
    task_routes={
        "app.worker.monte_carlo_tasks.*": {"queue": "monte_carlo"},
        # Add other routing rules here
    },
    
    # Redis visibility timeout (time until a task is visible again if a worker crashes)
    broker_transport_options={"visibility_timeout": 43200},  # 12 hours
    
    # Retry settings
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-expired-simulations": {
            "task": "app.worker.monte_carlo_tasks.cleanup_expired_simulations",
            "schedule": crontab(hour=2, minute=0),  # 2:00 AM every day
            "options": {"queue": "monte_carlo"},
        },
        # Add other periodic tasks here
    },
)

# Setup custom logging
@setup_logging.connect
def setup_celery_logging(**kwargs):
    """
    Configure logging for Celery
    """
    import logging
    from logging.config import dictConfig
    
    # Define log configuration
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s [%(process)d] [%(levelname)s] [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "detailed",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": os.path.join(settings.LOG_DIR, "celery.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 10,
            },
        },
        "loggers": {
            "celery": {
                "handlers": ["console", "file"],
                "level": settings.LOG_LEVEL,
                "propagate": False,
            },
            "app": {
                "handlers": ["console", "file"],
                "level": settings.LOG_LEVEL,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": settings.LOG_LEVEL,
        },
    }
    
    # Apply log configuration
    dictConfig(log_config)

# This allows the worker to be started with: celery -A app.worker.celery_app worker
if __name__ == "__main__":
    celery_app.start()

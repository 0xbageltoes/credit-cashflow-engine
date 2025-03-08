import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from app.core.config import settings
import logging
import sys
import json
from pythonjsonlogger import jsonlogger

# Initialize Sentry SDK
def init_sentry():
    # Skip Sentry initialization in test environment
    if settings.ENVIRONMENT == "test":
        logging.info("Skipping Sentry initialization in test environment")
        return
        
    if settings.SENTRY_DSN and settings.SENTRY_DSN != "https://your-key@sentry.io/your-project-id":
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.ENVIRONMENT,
            release=settings.VERSION,
            traces_sample_rate=0.2,
            integrations=[
                FastApiIntegration(),
                RedisIntegration(),
            ],
            # Performance tracing
            _experiments={
                "profiles_sample_rate": 0.1,
            }
        )

# Configure JSON logging for structured logs
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = record.created
        log_record['level'] = record.levelname
        log_record['service'] = 'credit-cashflow-engine'
        log_record['environment'] = settings.ENVIRONMENT
        log_record['version'] = settings.VERSION
        
        # Add trace context for distributed tracing
        if hasattr(sentry_sdk, 'Hub'):
            scope = sentry_sdk.Hub.current.scope
            if scope and scope.span:
                log_record['trace_id'] = scope.span.trace_id
                log_record['span_id'] = scope.span.span_id

def setup_logging():
    """Configure logging for the application"""
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    
    # Set up console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Set up JSON formatter
    json_formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    console_handler.setFormatter(json_formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    
    # Set up file handler for production
    if settings.ENVIRONMENT == 'production':
        file_handler = logging.FileHandler('logs/app.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    
    return root_logger

# Function to capture exceptions and log them
def log_exception(exc, context=None):
    """Log exception and capture with Sentry"""
    logger = logging.getLogger(__name__)
    
    error_details = {
        'exception_type': type(exc).__name__,
        'exception_message': str(exc),
    }
    
    if context:
        error_details['context'] = context
    
    logger.error(f"Exception occurred: {exc}", extra=error_details)
    
    # Capture exception with Sentry
    if settings.SENTRY_DSN:
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            sentry_sdk.capture_exception(exc)

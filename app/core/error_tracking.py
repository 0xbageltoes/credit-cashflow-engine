import logging
import sys
import json
from pythonjsonlogger import jsonlogger

# Conditionally import Sentry SDK
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.redis import RedisIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    logging.warning("Sentry SDK not installed, error tracking disabled")

from app.core.config import settings

# Initialize Sentry SDK
def init_sentry():
    # Skip Sentry initialization in test environment or if Sentry SDK is not available
    if settings.ENVIRONMENT == "test" or not SENTRY_AVAILABLE:
        logging.info("Skipping Sentry initialization in test environment or Sentry SDK not available")
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
        """Add fields from the record to the message dict."""
        super().add_fields(log_record, record, message_dict)

        # Import sentry_sdk locally to avoid NameError when not installed or available
        SENTRY_AVAILABLE = False
        try:
            import sentry_sdk
            SENTRY_AVAILABLE = True
        except ImportError:
            pass

        # Only try to use Sentry if it's available
        if SENTRY_AVAILABLE and hasattr(sentry_sdk, 'Hub'):
            # Get the current Sentry scope to access context data
            hub = sentry_sdk.Hub.current
            if hub.client:
                # Add trace ID if available
                scope = hub.scope
                if scope.span:
                    span_id = scope.span.span_id
                    trace_id = scope.span.trace_id
                    if span_id:
                        log_record['span_id'] = span_id
                    if trace_id:
                        log_record['trace_id'] = trace_id

        log_record['timestamp'] = record.created
        log_record['level'] = record.levelname
        log_record['service'] = 'credit-cashflow-engine'
        log_record['environment'] = settings.ENVIRONMENT
        log_record['version'] = settings.VERSION
        
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
    if settings.SENTRY_DSN and SENTRY_AVAILABLE:
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            sentry_sdk.capture_exception(exc)

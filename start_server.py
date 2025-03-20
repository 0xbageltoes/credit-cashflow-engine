"""
Start script for the Credit Cashflow Engine API server.

This script provides proper error handling and diagnostic information
when starting the FastAPI application.
"""
import os
import sys
import traceback
import logging
import asyncio
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("server_startup")

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import fastapi
        import sqlalchemy
        import pydantic
        
        logger.info(f"FastAPI version: {fastapi.__version__}")
        logger.info(f"SQLAlchemy version: {sqlalchemy.__version__}")
        logger.info(f"Pydantic version: {pydantic.__version__}")
        
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def check_environment():
    """Check if environment is properly configured."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_file):
        logger.warning(f".env file not found at {env_file}. Using environment variables or defaults.")
    else:
        logger.info(f".env file found at {env_file}")
    
    # Check for critical environment variables
    critical_vars = ['DATABASE_URL', 'SECRET_KEY']
    for var in critical_vars:
        if not os.environ.get(var):
            logger.warning(f"Environment variable {var} not set")

def start_server():
    """Start the FastAPI server with proper error handling."""
    try:
        logger.info("Starting server startup checks...")
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Missing dependencies. Please install required packages.")
            return 1
        
        # Check environment
        check_environment()
        
        # Import the app to check for any import errors
        logger.info("Importing application...")
        try:
            from app.main import app
            logger.info("Application imported successfully")
        except Exception as e:
            logger.error(f"Error importing application: {e}")
            traceback.print_exc()
            return 1
        
        # Start the server
        logger.info("Starting server on http://127.0.0.1:8000")
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="debug"
        )
        
        return 0
    except Exception as e:
        logger.error(f"Unhandled exception during server startup: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(start_server())

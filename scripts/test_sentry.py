"""
Test script to verify Sentry integration is working
"""
import sentry_sdk
import os
import sys
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env.test
env_path = Path(__file__).parent.parent / ".env.test"
load_dotenv(dotenv_path=env_path)

# Make sure we can access the app modules
sys.path.append(str(Path(__file__).parent.parent))

# After setting environment variables, import settings
from app.core.config import settings
from app.core.error_tracking import init_sentry

def main():
    print("Testing Sentry integration...")
    
    # Print the Sentry DSN (masked for security)
    dsn = settings.SENTRY_DSN
    if dsn:
        masked_dsn = f"{dsn[:20]}...{dsn[-10:]}" if len(dsn) > 30 else dsn
        print(f"Sentry DSN: {masked_dsn}")
    else:
        print("ERROR: Sentry DSN not found in environment variables!")
        return
    
    # Initialize Sentry
    print("Initializing Sentry...")
    init_sentry()
    
    try:
        # Generate a test error
        print("Generating a test message to send to Sentry...")
        sentry_sdk.capture_message("Test message from credit-cashflow-engine")
        
        # Give Sentry some time to send the message
        print("Waiting for Sentry to process the message...")
        time.sleep(1)
        
        # Generate a test exception
        print("Generating a test exception to send to Sentry...")
        raise ValueError("This is a test exception to verify Sentry is working")
    except Exception as e:
        # Capture the exception
        sentry_sdk.capture_exception(e)
        print(f"Test exception generated: {str(e)}")
        
        # Give Sentry some time to send the event
        print("Waiting for Sentry to process the exception...")
        time.sleep(2)
        
    print("\nCheck your Sentry dashboard at https://sentry.io to confirm the test events were received.")
    print("You should see both a test message and a test exception.")

if __name__ == "__main__":
    main()

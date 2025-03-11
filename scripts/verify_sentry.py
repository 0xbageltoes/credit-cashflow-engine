"""
Simple script to verify Sentry integration is working by causing a division by zero error
"""
import sentry_sdk
from pathlib import Path
from dotenv import load_dotenv
import time
import sys

# Load environment variables from .env.test
env_path = Path(__file__).parent.parent / ".env.test"
load_dotenv(dotenv_path=env_path)

# Get the DSN directly from environment variable
import os
sentry_dsn = os.getenv("SENTRY_DSN")
print(f"Initializing Sentry with DSN: {sentry_dsn}")

# Initialize Sentry directly with the DSN
sentry_sdk.init(
    dsn=sentry_dsn,
    # Set traces_sample_rate to 1.0 to capture 100% of transactions for performance monitoring
    traces_sample_rate=1.0,
    # Enable profiling (optional)
    _experiments={
        "profiles_sample_rate": 1.0,
    },
)

def main():
    print("Sending test error to Sentry...")
    try:
        # This is the exact example that Sentry recommends for verification
        division_by_zero = 1 / 0
    except Exception as e:
        print(f"Generated exception: {type(e).__name__}: {str(e)}")
        sentry_sdk.capture_exception(e)
        print("Waiting for Sentry to process the event...")
        # Give time for the event to be sent
        time.sleep(3)
        
    print("\nCheck your Sentry dashboard to confirm the test event was received.")
    print("You should see a ZeroDivisionError exception.")

if __name__ == "__main__":
    main()

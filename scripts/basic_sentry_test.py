"""
Basic Sentry integration test - following official documentation
"""
# Import the Sentry SDK
import sentry_sdk
import time

# Your DSN can be found in Sentry.io under Settings > Projects > Client Keys
SENTRY_DSN = "https://afff02ae0e7587659c7252b4678afd34@o4508954818510848.ingest.us.sentry.io/4508955339784192"

def main():
    print(f"Initializing Sentry with DSN: {SENTRY_DSN}")
    
    # Initialize the SDK
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        # Set traces_sample_rate to 1.0 to capture 100% of transactions for performance monitoring
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100% of sampled transactions
        profiles_sample_rate=1.0,
        # Enable performance monitoring
        enable_tracing=True,
    )
    
    print("Sentry SDK initialized.")
    
    try:
        print("Attempting division by zero...")
        1 / 0
    except Exception as e:
        print(f"Captured exception: {type(e).__name__}: {e}")
        
        # Explicitly capture the exception
        sentry_sdk.capture_exception(e)
        print("Exception captured by Sentry")
        
        # Also send a message
        sentry_sdk.capture_message("This is a test message from credit-cashflow-engine")
        print("Test message sent to Sentry")
    
    # Make sure events are sent before the script exits
    print("Waiting for events to be delivered to Sentry (5 seconds)...")
    time.sleep(5)
    
    print("Done! Check your Sentry dashboard to see if events were received.")

if __name__ == "__main__":
    main()

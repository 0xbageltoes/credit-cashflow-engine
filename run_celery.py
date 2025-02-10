import subprocess
import sys
import os
from pathlib import Path

def run_celery():
    # Get the directory containing this script
    current_dir = Path(__file__).parent.absolute()
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(current_dir)
    
    # Start Celery worker
    worker_command = [
        sys.executable, "-m", "celery",
        "-A", "app.core.celery_app",
        "worker",
        "--loglevel=info",
        "-Q", "forecasting,risk_analysis,market_data"
    ]
    
    worker = subprocess.Popen(worker_command, env=env)
    
    # Start Flower (monitoring)
    flower_command = [
        sys.executable, "-m", "celery",
        "-A", "app.core.celery_app",
        "flower"
    ]
    
    flower = subprocess.Popen(flower_command, env=env)
    
    try:
        worker.wait()
        flower.wait()
    except KeyboardInterrupt:
        print("\nShutting down Celery worker and Flower...")
        worker.terminate()
        flower.terminate()
        worker.wait()
        flower.wait()
        print("Shutdown complete")

if __name__ == "__main__":
    run_celery()

"""
Setup script for local AbsBox development without Docker

This script will:
1. Install the required dependencies
2. Verify the AbsBox installation
3. Set up a mock Hastructure engine for local development
"""
import subprocess
import sys
import os
import time
import json
from pathlib import Path
import threading
import http.server
import socketserver
from typing import Dict, Any

def print_header(text: str) -> None:
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80 + "\n")

def install_dependencies() -> bool:
    """Install required dependencies"""
    print_header("Installing Dependencies")
    
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        requirements_file = project_root / "requirements.txt"
        
        print(f"Installing from {requirements_file}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        
        # Install AbsBox separately to ensure we get the latest version
        print("Installing AbsBox...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "absbox", "--upgrade"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def verify_absbox_installation() -> bool:
    """Verify that AbsBox is installed correctly"""
    print_header("Verifying AbsBox Installation")
    
    try:
        # Import AbsBox
        import absbox as ab
        from absbox.local.pool import Pool
        from absbox.local.loan import FixedRateLoan
        from absbox.local.engine import LiqEngine
        
        print(f"AbsBox version: {ab.__version__}")
        
        # Create a simple test loan
        loan = FixedRateLoan(balance=100000.0, rate=0.05, originTerm=360, remainTerm=360)
        
        # Create a pool with the loan
        pool = Pool(assets=[loan])
        pool.setName("Test Pool")
        
        # Run the pool using the local engine
        engine = LiqEngine()
        result = engine.runPool(pool)
        
        # Check that we got a valid result
        if result and hasattr(result, "cashflow"):
            cashflow = result.cashflow()
            print(f"Successfully generated cashflows with {len(cashflow)} periods")
            print(f"First few cashflow periods: {cashflow.head(3)}")
            print("AbsBox installation verified successfully.")
            return True
        else:
            print("Error: AbsBox installation verification failed. No valid result returned.")
            return False
    except ImportError as e:
        print(f"Error importing AbsBox: {e}")
        print("Please ensure that AbsBox is installed correctly.")
        return False
    except Exception as e:
        print(f"Error verifying AbsBox installation: {e}")
        return False

class MockHastructureHandler(http.server.SimpleHTTPRequestHandler):
    """Handler for the mock Hastructure engine"""
    
    def _send_response(self, status: int, data: Dict[str, Any]) -> None:
        """Send a JSON response"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
    
    def do_GET(self) -> None:
        """Handle GET requests"""
        if self.path == "/health":
            # Health check endpoint
            self._send_response(200, {
                "status": "healthy",
                "version": "mock-1.0.0",
                "uptime": "0h 0m 0s",
                "workers": 1
            })
        else:
            # Unknown endpoint
            self._send_response(404, {
                "error": f"Unknown endpoint: {self.path}"
            })
    
    def do_POST(self) -> None:
        """Handle POST requests"""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        
        try:
            request_body = json.loads(post_data)
            
            if self.path == "/pool":
                # Pool calculation endpoint
                self._send_response(200, {
                    "status": "success",
                    "message": "Pool calculation completed",
                    "runTime": 0.5,
                    "cashflow": [
                        {"period": 1, "principal": 100.0, "interest": 50.0},
                        {"period": 2, "principal": 101.0, "interest": 49.0}
                    ]
                })
            elif self.path == "/deal":
                # Deal calculation endpoint
                self._send_response(200, {
                    "status": "success",
                    "message": "Deal calculation completed",
                    "runTime": 1.0,
                    "bondFlow": [
                        {"period": 1, "A": 70.0, "B": 30.0},
                        {"period": 2, "A": 71.0, "B": 29.0}
                    ],
                    "poolFlow": [
                        {"period": 1, "principal": 100.0, "interest": 50.0},
                        {"period": 2, "principal": 101.0, "interest": 49.0}
                    ]
                })
            else:
                # Unknown endpoint
                self._send_response(404, {
                    "error": f"Unknown endpoint: {self.path}"
                })
        except json.JSONDecodeError:
            self._send_response(400, {
                "error": "Invalid JSON body"
            })
        except Exception as e:
            self._send_response(500, {
                "error": f"Server error: {str(e)}"
            })
    
    def log_message(self, format, *args) -> None:
        """Override the logging to be more informative"""
        print(f"[MockHastructure] {self.address_string()} - {args[0]} {args[1]} {args[2]}")

def start_mock_hastructure() -> None:
    """Start a mock Hastructure engine"""
    print_header("Starting Mock Hastructure Engine")
    
    # Default port
    port = 8081
    handler = MockHastructureHandler
    
    # Create the server
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Mock Hastructure engine started at http://localhost:{port}")
        print("This provides a simple mock implementation of the Hastructure API.")
        print("Press Ctrl+C to stop.")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down mock Hastructure engine...")
            httpd.server_close()

def update_env_file() -> None:
    """Update the .env.test file with Hastructure settings"""
    print_header("Updating Environment Configuration")
    
    try:
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env.test"
        
        # Check if file exists
        if not env_file.exists():
            print(f"Warning: {env_file} does not exist. Creating it...")
            env_file.touch()
        
        # Read existing content
        content = env_file.read_text()
        
        # Add or update Hastructure settings
        hastructure_settings = """
# AbsBox and Hastructure Settings
HASTRUCTURE_URL=http://localhost:8081
HASTRUCTURE_TIMEOUT=300
HASTRUCTURE_MAX_POOL_SIZE=1
"""
        
        # Check if settings already exist
        if "HASTRUCTURE_URL" not in content:
            # Append settings
            with env_file.open("a") as f:
                f.write(hastructure_settings)
            print(f"Added Hastructure settings to {env_file}")
        else:
            print(f"Hastructure settings already exist in {env_file}")
        
        print("Environment configuration updated successfully.")
    except Exception as e:
        print(f"Error updating environment configuration: {e}")

def main() -> None:
    """Main function"""
    print_header("AbsBox Local Development Setup")
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Exiting.")
        return
    
    # Verify AbsBox installation
    if not verify_absbox_installation():
        print("Failed to verify AbsBox installation. Exiting.")
        return
    
    # Update environment configuration
    update_env_file()
    
    # Start mock Hastructure engine in a separate thread
    print("\nSetup completed successfully!")
    print("You can now run the mock Hastructure engine with:")
    print("    python scripts/setup_local_absbox.py --mock-server")
    
    # Check if we should start the mock server
    if "--mock-server" in sys.argv:
        # Start the mock server
        start_mock_hastructure()

if __name__ == "__main__":
    main()

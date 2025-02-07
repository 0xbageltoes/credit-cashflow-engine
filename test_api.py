import httpx
import json
from datetime import datetime, timedelta
import sys

# Test data
test_data = {
    "loans": [
        {
            "principal": 100000,
            "interest_rate": 0.05,  # 5% annual rate
            "term_months": 360,     # 30-year loan
            "start_date": (datetime.now()).isoformat(),
            "prepayment_assumption": 0.02  # 2% annual prepayment rate
        }
    ],
    "scenario_name": "Test Scenario",
    "monte_carlo_sims": None,
    "assumptions": {
        "default_rate": 0.01
    }
}

# API endpoint
base_url = "http://localhost:8000"

# Your Supabase JWT token
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsImtpZCI6Ijg4UmY5V3hXNXFIbFVSTjQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3ZzenFzZm50Y3FpZGdoY3d4ZWlqLnN1cGFiYXNlLmNvL2F1dGgvdjEiLCJzdWIiOiJmNTEzNGY2Mi0wYzJiLTQ2OWMtOWFkYi1iNDk0OThiZWI0NDgiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzM4ODkxODM2LCJpYXQiOjE3Mzg4ODgyMzYsImVtYWlsIjoiYWxlYW5vc0BtZS5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImVtYWlsIiwicHJvdmlkZXJzIjpbImVtYWlsIl19LCJ1c2VyX21ldGFkYXRhIjp7ImVtYWlsIjoiYWxlYW5vc0BtZS5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJzdWIiOiJmNTEzNGY2Mi0wYzJiLTQ2OWMtOWFkYi1iNDk0OThiZWI0NDgifSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJwYXNzd29yZCIsInRpbWVzdGFtcCI6MTczODg4ODIzNn1dLCJzZXNzaW9uX2lkIjoiNjM0YzcwOGYtYzIwZi00ODQ5LTkyNGMtMTEwODFkM2RhZmMxIiwiaXNfYW5vbnltb3VzIjpmYWxzZX0.JUfGxwJeYpQF2QWW8C5TR2lTnglkSElfMNlqHnZ7sVM",
    "Content-Type": "application/json"
}

def test_forecast_endpoint():
    """Test the cash flow forecast endpoint"""
    print("Testing /cashflow/forecast endpoint...")
    print(f"Sending request to {base_url}/cashflow/forecast")
    print(f"Request data: {json.dumps(test_data, indent=2)}")
    
    try:
        print("Making HTTP request...")
        client = httpx.Client(timeout=120.0)  # 2 minute timeout
        response = client.post(
            f"{base_url}/cashflow/forecast",
            json=test_data,
            headers=headers
        )
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            print("Forecast successful!")
            result = response.json()
            print(f"Response data: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response body: {response.text}")
            
    except httpx.TimeoutException as e:
        print(f"Request timed out after {e.request.timeout} seconds")
        print(f"Error type: {type(e)}")
        print("Traceback:", file=sys.stderr)
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e)}")
        print("Traceback:", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    test_forecast_endpoint()

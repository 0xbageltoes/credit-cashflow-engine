import httpx
import json
import asyncio
import websockets
from datetime import datetime, timedelta
import sys

# Test data
test_data = {
    "loans": [
        {
            "principal": 100000.0,
            "interest_rate": 0.05,  # 5% annual rate
            "term_months": 360,     # 30-year loan
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "prepayment_assumption": 0.02  # 2% annual prepayment rate
        }
    ],
    "discount_rate": 0.03,     # 3% discount rate
    "run_monte_carlo": True,
    "monte_carlo_config": {
        "num_simulations": 1000,
        "default_prob": 0.02,
        "prepay_prob": 0.05,
        "rate_volatility": 0.01
    },
    "scenario_name": "Test Scenario",
    "assumptions": {
        "default_rate": 0.01,
        "recovery_rate": 0.6
    }
}

# API endpoint
base_url = "http://localhost:8000/api/v1/forecasting"

# Your Supabase JWT token
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsImtpZCI6Ijg4UmY5V3hXNXFIbFVSTjQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3ZzenFzZm50Y3FpZGdoY3d4ZWlqLnN1cGFiYXNlLmNvL2F1dGgvdjEiLCJzdWIiOiJmNTEzNGY2Mi0wYzJiLTQ2OWMtOWFkYi1iNDk0OThiZWI0NDgiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzM4OTIyOTg2LCJpYXQiOjE3Mzg5MTkzODYsImVtYWlsIjoiYWxlYW5vc0BtZS5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImVtYWlsIiwicHJvdmlkZXJzIjpbImVtYWlsIl19LCJ1c2VyX21ldGFkYXRhIjp7ImVtYWlsIjoiYWxlYW5vc0BtZS5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJzdWIiOiJmNTEzNGY2Mi0wYzJiLTQ2OWMtOWFkYi1iNDk0OThiZWI0NDgifSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJwYXNzd29yZCIsInRpbWVzdGFtcCI6MTczODkxOTM4Nn1dLCJzZXNzaW9uX2lkIjoiY2VmODZjNDAtMWFkMy00Nzg2LThiOTQtMDljYzQ1MjI2MDFkIiwiaXNfYW5vbnltb3VzIjpmYWxzZX0.ILFUjPdMmYp11S-Bf-fFGN_-aYa-QZ1Mq_TNeEPShHs",
    "Content-Type": "application/json"
}

async def test_async_forecast():
    """Test the async forecast endpoint with WebSocket updates"""
    print("Testing async forecast with WebSocket updates...")
    print(f"Request data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Start forecast calculation
        async with httpx.AsyncClient() as client:
            print("Queuing forecast calculation...")
            response = await client.post(
                f"{base_url}/forecast/async",
                json=test_data,
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"Error queuing forecast: {response.status_code}")
                print(f"Response: {response.text}")
                return
            
            task_id = response.json()["task_id"]
            print(f"Task ID: {task_id}")
            
            # Connect to WebSocket for updates
            user_id = "f5134f62-0c2b-469c-9adb-b49498beb448"  # From JWT token
            ws_url = f"ws://localhost:8000/api/v1/forecasting/ws/{user_id}"
            
            async with websockets.connect(ws_url) as websocket:
                # Subscribe to task updates
                await websocket.send(json.dumps({
                    "type": "subscribe",
                    "task_id": task_id
                }))
                
                print("Waiting for updates...")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    print(f"Received update: {json.dumps(data, indent=2)}")
                    
                    # Check task status
                    status_response = await client.get(
                        f"{base_url}/forecast/{task_id}/status",
                        headers=headers
                    )
                    status = status_response.json()["status"]
                    print(f"Task status: {status}")
                    
                    if status == "completed" or status.startswith("failed"):
                        break
                    
                    await asyncio.sleep(1)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Traceback:", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_async_forecast())

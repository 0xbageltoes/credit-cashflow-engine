"""
Locust Load Test File for Credit Cashflow Engine API
"""

import json
import time
from locust import HttpUser, task, between
from locust.exception import RescheduleTask


class CashflowAPIUser(HttpUser):
    """Simulates users of the cashflow API"""
    
    # Wait 1-5 seconds between tasks
    wait_time = between(1, 5)
    
    def on_start(self):
        """Setup before tests - authenticate if needed"""
        # For real authentication with Supabase, we'd do something like:
        # response = self.client.post("/api/auth/login", 
        #     json={"email": "test@example.com", "password": "password"})
        # self.token = response.json()["access_token"]
        
        # For testing, we'll assume a mock token is accepted
        self.token = "mock_test_token"
        self.headers = {"Authorization": f"Bearer {self.token}"}
        
        # Sample loan for cashflow calculation
        self.sample_loan = {
            "principal": 100000,
            "rate": 0.05,
            "term": 360,
            "start_date": "2025-01-01"
        }
        
        # Sample for batch processing
        self.batch_loans = {
            "loans": [
                {
                    "principal": 100000,
                    "rate": 0.05,
                    "term": 360,
                    "start_date": "2025-01-01"
                },
                {
                    "principal": 250000,
                    "rate": 0.04,
                    "term": 240,
                    "start_date": "2025-02-01"
                },
                {
                    "principal": 50000,
                    "rate": 0.035,
                    "term": 180,
                    "start_date": "2025-03-01"
                }
            ]
        }
    
    @task(2)
    def get_health(self):
        """Check API health - basic endpoint that should always work"""
        with self.client.get("/api/v1/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(3)
    def calculate_single_loan(self):
        """Calculate cashflow for a single loan"""
        with self.client.post(
            "/api/v1/cashflow/calculate", 
            headers=self.headers,
            json=self.sample_loan,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                # Verify response structure
                data = response.json()
                if "cashflows" not in data:
                    response.failure("Response missing cashflows data")
            elif response.status_code == 401:
                # Authentication issues - we'll just log this in testing
                response.failure("Authentication failed")
            elif response.status_code == 429:
                # Rate limiting - reschedule the task
                response.failure("Rate limited")
                raise RescheduleTask()
            else:
                response.failure(f"Request failed with status {response.status_code}")
    
    @task(1)
    def calculate_batch(self):
        """Calculate cashflow for multiple loans in batch"""
        with self.client.post(
            "/api/v1/cashflow/calculate-batch", 
            headers=self.headers,
            json=self.batch_loans,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                # Verify response structure
                data = response.json()
                if "results" not in data:
                    response.failure("Response missing results data")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            elif response.status_code == 429:
                response.failure("Rate limited")
                raise RescheduleTask()
            else:
                response.failure(f"Request failed with status {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """Get API metrics - typically admin only"""
        with self.client.get(
            "/api/v1/metrics", 
            headers=self.headers,
            catch_response=True
        ) as response:
            # This might return 403 if not authorized, which is acceptable
            if response.status_code not in [200, 403]:
                response.failure(f"Metrics request failed with status {response.status_code}")


class CashflowUIUser(HttpUser):
    """
    Simulates users accessing the API through a UI
    Has different patterns - more read operations and longer pauses
    """
    
    wait_time = between(3, 8)  # UI users have longer think time
    
    def on_start(self):
        """Setup before tests"""
        self.token = "mock_test_token_ui_user"
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.scenario_id = "mock-scenario-id-12345"
    
    @task(4)
    def get_scenarios(self):
        """Get list of scenarios - common read operation"""
        with self.client.get(
            "/api/v1/scenarios", 
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed to get scenarios: {response.status_code}")
    
    @task(2)
    def get_scenario_detail(self):
        """Get details of a specific scenario"""
        with self.client.get(
            f"/api/v1/scenarios/{self.scenario_id}", 
            headers=self.headers,
            catch_response=True
        ) as response:
            # 404 is acceptable if scenario doesn't exist
            if response.status_code not in [200, 404]:
                response.failure(f"Failed to get scenario detail: {response.status_code}")
    
    @task(1)
    def create_scenario(self):
        """Create a new scenario - less frequent operation"""
        scenario_data = {
            "name": f"Test Scenario {time.time()}",
            "description": "Created during load testing",
            "forecast_request": {
                "loans": [
                    {
                        "principal": 150000,
                        "rate": 0.045,
                        "term": 300,
                        "start_date": "2025-04-01"
                    }
                ]
            }
        }
        
        with self.client.post(
            "/api/v1/scenarios", 
            headers=self.headers,
            json=scenario_data,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                # Store ID for future requests
                data = response.json()
                if "id" in data:
                    self.scenario_id = data["id"]
            else:
                response.failure(f"Failed to create scenario: {response.status_code}")

from locust import HttpUser, task, between, constant_pacing, events, tag
import random
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global test configuration
TEST_CONFIG = {
    "use_fixed_token": True,
    "fixed_token": "YOUR_JWT_TOKEN_HERE",  # Replace with actual token for testing
    "api_version": "v1",  # API version prefix
    "user_count": 10,  # Number of simulated users
    "ramp_up_time": 30,  # Seconds to ramp up load
    "run_time": 300,  # Total test duration in seconds
    "log_requests": False,  # Enable for detailed request logging
    "scenarios_count": 5,  # Number of scenarios to create
}

# Loan templates for different types of loans
LOAN_TEMPLATES = {
    "mortgage": {
        "principal": lambda: random.uniform(100000, 500000),
        "interest_rate": lambda: random.uniform(0.03, 0.06),
        "term_months": lambda: random.choice([180, 240, 360]),
        "rate_type": "fixed",
        "prepayment_assumption": lambda: random.uniform(0.01, 0.03),
    },
    "auto_loan": {
        "principal": lambda: random.uniform(20000, 50000),
        "interest_rate": lambda: random.uniform(0.04, 0.08),
        "term_months": lambda: random.choice([36, 48, 60, 72]),
        "rate_type": "fixed",
        "prepayment_assumption": lambda: random.uniform(0.02, 0.05),
    },
    "personal_loan": {
        "principal": lambda: random.uniform(5000, 25000),
        "interest_rate": lambda: random.uniform(0.06, 0.15),
        "term_months": lambda: random.choice([12, 24, 36, 48]),
        "rate_type": "fixed",
        "prepayment_assumption": lambda: random.uniform(0.02, 0.04),
    },
    "business_loan": {
        "principal": lambda: random.uniform(50000, 1000000),
        "interest_rate": lambda: random.uniform(0.045, 0.08),
        "term_months": lambda: random.choice([12, 24, 36, 48, 60]),
        "rate_type": lambda: random.choice(["fixed", "variable"]),
        "prepayment_assumption": lambda: random.uniform(0.01, 0.05),
    },
    "credit_line": {
        "principal": lambda: random.uniform(10000, 100000),
        "interest_rate": lambda: random.uniform(0.07, 0.17),
        "term_months": lambda: random.choice([12, 24, 36]),
        "rate_type": "variable",
        "prepayment_assumption": lambda: random.uniform(0.03, 0.08),
    }
}

def generate_random_loan():
    """Generate a random loan using the templates"""
    loan_type = random.choice(list(LOAN_TEMPLATES.keys()))
    template = LOAN_TEMPLATES[loan_type]
    
    # Generate a start date between 90 days ago and today
    days_ago = random.randint(0, 90)
    start_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    loan = {
        "loan_type": loan_type,
        "principal": template["principal"](),
        "interest_rate": template["interest_rate"](),
        "term_months": template["term_months"](),
        "start_date": start_date,
        "prepayment_assumption": template["prepayment_assumption"](),
        "rate_type": template["rate_type"]() if callable(template["rate_type"]) else template["rate_type"]
    }
    
    # Add some variability for variable rate loans
    if loan["rate_type"] == "variable":
        loan["rate_adjustment_period"] = random.choice([6, 12])
        loan["rate_cap"] = loan["interest_rate"] + random.uniform(0.01, 0.03)
        loan["rate_floor"] = max(0.01, loan["interest_rate"] - random.uniform(0.01, 0.02))
    
    return loan

def generate_forecast_payload(loan_count=None, run_monte_carlo=False):
    """Generate a realistic forecast payload with multiple loans"""
    if loan_count is None:
        loan_count = random.randint(1, 5)
    
    loans = [generate_random_loan() for _ in range(loan_count)]
    
    # Generate sensible discount rate (usually close to weighted average interest rate)
    avg_interest = sum(loan["interest_rate"] for loan in loans) / len(loans)
    discount_rate = max(0.01, avg_interest - random.uniform(0.005, 0.015))
    
    payload = {
        "loans": loans,
        "discount_rate": discount_rate,
        "run_monte_carlo": run_monte_carlo,
        "forecast_months": random.choice([12, 24, 36, 60, 120, 360]),
    }
    
    # Add Monte Carlo specific parameters if needed
    if run_monte_carlo:
        payload["monte_carlo_iterations"] = 100
        payload["monte_carlo_confidence_interval"] = 0.95
    
    return payload

def api_path(path):
    """Format API path with version prefix"""
    return f"/api/{TEST_CONFIG['api_version']}/{path.lstrip('/')}"

# Event hooks for statistics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info(f"Load test starting with {TEST_CONFIG['user_count']} users")
    logger.info(f"API version: {TEST_CONFIG['api_version']}")
    environment.runner.stats.reset_all()

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    if TEST_CONFIG["log_requests"] and exception is None:
        logger.info(f"Request {name} completed in {response_time}ms")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("Load test completed")

class CashflowUser(HttpUser):
    """Simulates a user of the cashflow forecasting API"""
    wait_time = between(1, 5)
    
    # User-specific state
    token = None
    headers = None
    scenario_ids = []
    user_profile = None
    
    def on_start(self):
        """Setup for each user"""
        self.user_id = f"loadtest_user_{random.randint(1000, 9999)}"
        
        if TEST_CONFIG["use_fixed_token"]:
            self.token = TEST_CONFIG["fixed_token"]
        else:
            # In a real environment, we would authenticate here
            # This is mocked for the load test
            self.token = f"simulated_token_{self.user_id}"
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "X-User-ID": self.user_id,
            "X-Request-ID": f"req_{int(time.time())}_{random.randint(1000, 9999)}"
        }
        
        # Generate user profile data
        self.user_profile = {
            "user_type": random.choice(["individual", "business", "institutional"]),
            "risk_tolerance": random.choice(["low", "medium", "high"]),
            "portfolio_size": random.choice(["small", "medium", "large"]),
        }
        
        # Create some initial scenarios to use during the test
        self.create_initial_scenarios()
    
    def create_initial_scenarios(self):
        """Create some initial scenarios to use during the test"""
        for i in range(TEST_CONFIG["scenarios_count"]):
            scenario_name = f"Load Test Scenario {i+1} - {self.user_id}"
            payload = {
                "name": scenario_name,
                "description": f"Auto-generated scenario for load testing {i+1}",
                "forecast_params": generate_forecast_payload(loan_count=random.randint(1, 3))
            }
            
            response = self.client.post(
                api_path("/cashflow/scenario/save"),
                json=payload,
                headers=self.headers,
                name="[Initialize] Create Scenario"
            )
            
            # Store the scenario ID if creation was successful
            if response.status_code == 200 or response.status_code == 201:
                try:
                    data = response.json()
                    if "id" in data:
                        self.scenario_ids.append(data["id"])
                except:
                    pass
    
    @tag("health")
    @task(5)
    def check_health(self):
        """Check the API health endpoint - high frequency, low impact"""
        self.client.get(
            "/health", 
            name="Health Check"
        )
    
    @tag("forecast")
    @task(10)
    def forecast_cashflow(self):
        """Run a standard cash flow forecast - most common API operation"""
        # Use more realistic data
        payload = generate_forecast_payload(run_monte_carlo=False)
        
        # Update request headers with unique request ID
        headers = self.headers.copy()
        headers["X-Request-ID"] = f"req_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Make the request
        self.client.post(
            api_path("/cashflow/forecast"),
            json=payload,
            headers=headers,
            name="Cash Flow Forecast - Standard"
        )
    
    @tag("forecast")
    @task(2)
    def forecast_with_montecarlo(self):
        """Run a more intensive Monte Carlo simulation - less frequent, higher impact"""
        # Generate payload with Monte Carlo enabled
        payload = generate_forecast_payload(run_monte_carlo=True)
        
        # Update request headers with unique request ID
        headers = self.headers.copy()
        headers["X-Request-ID"] = f"req_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Make the request with a higher timeout
        response = self.client.post(
            api_path("/cashflow/forecast"),
            json=payload,
            headers=headers,
            name="Cash Flow Forecast - Monte Carlo",
            timeout=30
        )
    
    @tag("scenarios")
    @task(5)
    def get_scenarios(self):
        """Get all saved scenarios"""
        self.client.get(
            api_path("/cashflow/scenarios"),
            headers=self.headers,
            name="List All Scenarios"
        )
    
    @tag("scenarios")
    @task(3)
    def get_specific_scenario(self):
        """Get a specific saved scenario if available"""
        if self.scenario_ids:
            scenario_id = random.choice(self.scenario_ids)
            self.client.get(
                api_path(f"/cashflow/scenario/{scenario_id}"),
                headers=self.headers,
                name="Get Specific Scenario"
            )
    
    @tag("scenarios")
    @task(2)
    def save_scenario(self):
        """Save a new forecasting scenario"""
        scenario_name = f"Scenario {int(time.time())}"
        payload = {
            "name": scenario_name,
            "description": f"Scenario created during load test",
            "forecast_params": generate_forecast_payload()
        }
        
        response = self.client.post(
            api_path("/cashflow/scenario/save"),
            json=payload,
            headers=self.headers,
            name="Save New Scenario"
        )
        
        # Store the scenario ID if creation was successful
        if response.status_code == 200 or response.status_code == 201:
            try:
                data = response.json()
                if "id" in data:
                    self.scenario_ids.append(data["id"])
                    # Cap the number of scenarios we track to avoid memory issues
                    if len(self.scenario_ids) > 20:
                        self.scenario_ids = self.scenario_ids[-20:]
            except:
                pass
    
    @tag("history")
    @task(3)
    def get_history(self):
        """Get forecast history"""
        self.client.get(
            api_path("/cashflow/history"),
            headers=self.headers,
            name="Get Forecast History"
        )
    
    @tag("history")
    @task(1)
    def get_history_filtered(self):
        """Get filtered forecast history"""
        # Random date in the last 30 days
        days_ago = random.randint(1, 30)
        start_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        self.client.get(
            api_path(f"/cashflow/history?start_date={start_date}"),
            headers=self.headers,
            name="Get Filtered History"
        )
    
    @tag("long_running")
    @constant_pacing(60)  # Runs at most once per minute
    @task(1)
    def complex_scenario_analysis(self):
        """Run a complex analysis with many loans - less frequent, high impact"""
        # Generate a more complex payload with more loans
        payload = generate_forecast_payload(loan_count=random.randint(8, 15), run_monte_carlo=True)
        payload["detailed_analysis"] = True
        payload["sensitivities"] = {
            "interest_rate": [-0.02, -0.01, 0, 0.01, 0.02],
            "prepayment": [-0.01, 0, 0.01]
        }
        
        # Update request headers with unique request ID
        headers = self.headers.copy()
        headers["X-Request-ID"] = f"req_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Make the request with a higher timeout
        self.client.post(
            api_path("/cashflow/forecast"),
            json=payload,
            headers=headers,
            name="Complex Scenario Analysis",
            timeout=60
        )
    
    @tag("error_cases")  
    @task(1)
    def test_error_handling(self):
        """Test the error handling capabilities of the API"""
        # Deliberately send invalid data
        error_cases = [
            # Missing required field
            {"discount_rate": 0.03, "run_monte_carlo": False},
            # Invalid data type
            {"loans": "not_a_list", "discount_rate": 0.03, "run_monte_carlo": False},
            # Out of range values
            {"loans": [generate_random_loan()], "discount_rate": -0.5, "run_monte_carlo": False},
        ]
        
        error_case = random.choice(error_cases)
        self.client.post(
            api_path("/cashflow/forecast"),
            json=error_case,
            headers=self.headers,
            name="Error Case - Invalid Request",
            catch_response=True,
            timeout=10
        )


class CashflowAPILoadTest(HttpUser):
    """Simulates load testing specific API endpoints"""
    wait_time = between(0.1, 1)  # Fast requests
    
    def on_start(self):
        """Setup for the API load tester"""
        self.token = TEST_CONFIG["fixed_token"]
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "X-User-ID": "loadtest_api",
            "X-Request-ID": f"req_{int(time.time())}_{random.randint(1000, 9999)}"
        }
    
    @tag("load_test")
    @task(1)
    def test_api_under_load(self):
        """Hit the API with a simple request to test performance under load"""
        # Simple and fast request that will be executed frequently
        self.client.get(
            "/health",
            name="API Load Test - Health"
        )


# Advanced usage examples:
# 1. Run with specific user count and spawn rate:
#    locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 10 --run-time 5m
#
# 2. Run headless and output to CSV:
#    locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 10m --headless --csv=results
#
# 3. Run with specific tag:
#    locust -f locustfile.py --host=http://localhost:8000 --tags forecast,scenarios
#
# 4. Run distributed with multiple workers:
#    locust -f locustfile.py --master --expect-workers=4  # On master node
#    locust -f locustfile.py --worker --master-host=<master_ip>  # On worker nodes

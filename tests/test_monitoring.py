import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families
import time
from app.core.monitoring import PrometheusMetrics

@pytest.fixture
def prometheus_url():
    return "http://localhost:9090"

@pytest.fixture
def grafana_url():
    return "http://localhost:3000"

class TestMonitoring:
    """Test monitoring and metrics collection"""

    def test_prometheus_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = requests.get("http://localhost:8000/metrics")
        assert response.status_code == 200
        
        # Parse metrics
        metrics = list(text_string_to_metric_families(response.text))
        
        # Verify essential metrics exist
        metric_names = [m.name for m in metrics]
        assert "http_requests_total" in metric_names
        assert "http_request_duration_seconds" in metric_names
        assert "cashflow_calculations_total" in metric_names
        assert "monte_carlo_simulation_duration_seconds" in metric_names

    def test_prometheus_up(self, prometheus_url):
        """Test Prometheus is up and collecting metrics"""
        response = requests.get(f"{prometheus_url}/-/healthy")
        assert response.status_code == 200

    def test_grafana_up(self, grafana_url):
        """Test Grafana is up"""
        response = requests.get(f"{grafana_url}/api/health")
        assert response.status_code == 200

    def test_redis_metrics(self):
        """Test Redis metrics collection"""
        response = requests.get("http://localhost:9090/api/v1/query", params={
            "query": "redis_connected_clients"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["data"]["result"]) > 0

    def test_celery_metrics(self):
        """Test Celery metrics collection"""
        response = requests.get("http://localhost:9090/api/v1/query", params={
            "query": "celery_tasks_total"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

class TestPerformance:
    """Test performance and scaling"""

    @pytest.mark.benchmark
    def test_api_response_time(self, client, sample_loan_request):
        """Test API response time"""
        start_time = time.time()
        response = client.post("/api/v1/forecast", json=sample_loan_request)
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 2.0  # Response should be under 2 seconds

    @pytest.mark.benchmark
    def test_batch_processing_scaling(self, client):
        """Test batch processing performance scaling"""
        # Create requests with increasing batch sizes
        batch_sizes = [5, 10, 20]
        times = []
        
        for size in batch_sizes:
            request = {
                "forecasts": [
                    {
                        "loans": [{
                            "principal": 100000,
                            "interest_rate": 0.05,
                            "term_months": 360,
                            "start_date": "2025-01-01",
                            "prepayment_assumption": 0.02
                        }]
                    }
                ] * size,
                "parallel": True
            }
            
            start_time = time.time()
            response = client.post("/api/v1/forecast/batch", json=request)
            end_time = time.time()
            
            assert response.status_code == 200
            times.append(end_time - start_time)
        
        # Verify sub-linear scaling (time shouldn't increase linearly with batch size)
        assert times[1] / times[0] < 2.0  # Doubling batch size should take less than 2x time
        assert times[2] / times[1] < 2.0

    def test_redis_cache_performance(self, client, redis_cache, sample_loan_request):
        """Test Redis cache performance"""
        # Make initial request (uncached)
        start_time = time.time()
        response1 = client.post("/api/v1/forecast", json=sample_loan_request)
        uncached_time = time.time() - start_time
        
        # Make same request again (should be cached)
        start_time = time.time()
        response2 = client.post("/api/v1/forecast", json=sample_loan_request)
        cached_time = time.time() - start_time
        
        assert response1.json() == response2.json()
        assert cached_time < uncached_time * 0.5  # Cached response should be at least 2x faster

    def test_celery_worker_scaling(self, client):
        """Test Celery worker auto-scaling"""
        # Submit multiple async tasks
        task_ids = []
        for _ in range(5):
            response = client.post("/api/v1/forecast/async", json=sample_loan_request)
            assert response.status_code == 202
            task_ids.append(response.json()["task_id"])
        
        # Check Celery metrics for worker count
        response = requests.get("http://localhost:9090/api/v1/query", params={
            "query": "celery_workers"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        # Verify tasks complete
        for task_id in task_ids:
            for _ in range(30):  # Wait up to 30 seconds
                status_response = client.get(f"/api/v1/forecast/status/{task_id}")
                if status_response.json()["status"] == "completed":
                    break
                time.sleep(1)
            assert status_response.json()["status"] == "completed"

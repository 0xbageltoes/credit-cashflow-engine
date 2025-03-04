import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.core.monitoring import PrometheusMetrics

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def metrics():
    return PrometheusMetrics()

class TestMonitoring:
    """Test monitoring and metrics collection"""

    def test_metrics_collection(self, metrics):
        """Test metrics collection"""
        # Test request metrics
        metrics.track_request("/test", "GET", 200)
        assert metrics.request_count._value.get(("/test", "GET", "200")) == 1

        # Test cache metrics
        metrics.track_cache(hit=True)
        assert metrics.cache_hits._value.get() == 1
        metrics.track_cache(hit=False)
        assert metrics.cache_misses._value.get() == 1

    @patch("app.core.monitoring.time.time")
    def test_request_latency(self, mock_time, metrics):
        """Test request latency tracking"""
        mock_time.side_effect = [0, 1]  # Simulate 1 second elapsed

        @metrics.track_request_latency("/test")
        async def test_func():
            return "test"

        # Run the decorated function
        import asyncio
        result = asyncio.run(test_func())
        assert result == "test"

        # Verify latency was recorded
        assert len(metrics.request_latency._buckets) > 0

    def test_task_tracking(self, metrics):
        """Test task tracking"""
        @metrics.track_task("test_task")
        async def test_task():
            return "completed"

        # Run the task
        import asyncio
        result = asyncio.run(test_task())
        assert result == "completed"

        # Verify task metrics
        assert metrics.active_tasks._value.get(("test_task",)) == 0

    def test_system_info(self, metrics):
        """Test system info collection"""
        test_info = {
            "version": "1.0.0",
            "environment": "test",
            "python_version": "3.10.4"
        }
        metrics.update_system_info(test_info)
        assert metrics.system_info._value == test_info

class TestPerformance:
    """Test performance and scaling"""

    def test_api_response_time(self, client):
        """Test API response time"""
        start_time = pytest.importorskip("time").time()
        response = client.get("/health")
        end_time = pytest.importorskip("time").time()
        
        assert response.status_code == 200
        assert end_time - start_time < 1.0  # Response should be under 1 second

    @patch("app.core.redis_cache.RedisCache")
    def test_cache_performance(self, mock_redis, client):
        """Test cache performance"""
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True

        # Test cache operations
        start_time = pytest.importorskip("time").time()
        for _ in range(100):
            mock_redis.get("test_key")
            mock_redis.set("test_key", "test_value")
        end_time = pytest.importorskip("time").time()

        # Cache operations should be fast
        assert end_time - start_time < 1.0

    @patch("app.core.celery_app.celery")
    def test_task_queue_performance(self, mock_celery):
        """Test task queue performance"""
        mock_celery.send_task.return_value = MagicMock(id="test_task_id")

        # Test task submission
        start_time = pytest.importorskip("time").time()
        for _ in range(10):
            mock_celery.send_task("test_task")
        end_time = pytest.importorskip("time").time()

        # Task submission should be fast
        assert end_time - start_time < 0.5

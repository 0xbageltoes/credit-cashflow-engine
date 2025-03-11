"""
Integration tests for the AbsBox service with Redis caching.

These tests verify that the AbsBox service properly integrates with Redis
for caching structured finance analysis results.
"""
import os
import sys
import json
import time
import uuid
import pytest
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load test environment variables
from dotenv import load_dotenv
load_dotenv(".env.test")

# Set test environment variables
os.environ["ENVIRONMENT"] = "test"
os.environ["USE_MOCK_HASTRUCTURE"] = "true"
os.environ["ABSBOX_CACHE_ENABLED"] = "true"

# Import services
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced
from app.services.redis_client import RedisClient

# Import models
from app.models.structured_products import (
    StructuredDealRequest, 
    StructuredDealResponse,
    LoanPoolConfig,
    LoanConfig,
    WaterfallConfig,
    AccountConfig,
    BondConfig,
    WaterfallAction,
    ScenarioConfig,
    DefaultCurveConfig,
    PrepaymentCurveConfig
)

# Skip all tests if Redis is not available
pytestmark = pytest.mark.skipif(
    os.environ.get("REDIS_URL") is None,
    reason="Redis URL not configured for testing"
)

@pytest.fixture
def redis_client():
    """Fixture to get a Redis client"""
    try:
        client = RedisClient()
        # Verify connectivity
        assert client.ping(), "Redis ping failed"
        # Clear test namespace
        for key in client.keys("test:absbox:*"):
            client.delete(key)
        return client
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")

@pytest.fixture
def absbox_service():
    """Fixture to get an AbsBox service with cache enabled"""
    config = {
        "use_cache": True,
        "cache_ttl": 60,  # Short TTL for testing
        "use_mock": True
    }
    return AbsBoxServiceEnhanced(config)

@pytest.fixture
def sample_deal():
    """Fixture to create a sample structured deal request"""
    # Create a simple deal with a few loans and bonds
    loan1 = LoanConfig(
        balance=100000.0,
        rate=0.05,
        term=360,
        start_date="2023-01-01",
        rate_type="fixed",
        payment_frequency="Monthly"
    )
    
    loan2 = LoanConfig(
        balance=150000.0,
        rate=0.045,
        term=360,
        start_date="2023-01-15",
        rate_type="fixed",
        payment_frequency="Monthly"
    )
    
    pool = LoanPoolConfig(
        pool_name="Test Pool",
        loans=[loan1, loan2]
    )
    
    waterfall = WaterfallConfig(
        start_date="2023-02-01",
        accounts=[
            AccountConfig(
                name="ReserveFund",
                initial_balance=10000.0
            )
        ],
        bonds=[
            BondConfig(
                name="ClassA",
                balance=200000.0,
                rate=0.04
            ),
            BondConfig(
                name="ClassB",
                balance=50000.0,
                rate=0.06
            )
        ],
        actions=[
            WaterfallAction(
                source="CollectedInterest",
                target="ClassA",
                amount="Interest"
            ),
            WaterfallAction(
                source="CollectedInterest",
                target="ClassB",
                amount="Interest"
            ),
            WaterfallAction(
                source="CollectedPrincipal",
                target="ClassA",
                amount="OutstandingPrincipal"
            ),
            WaterfallAction(
                source="CollectedPrincipal",
                target="ClassB",
                amount="OutstandingPrincipal"
            )
        ]
    )
    
    scenario = ScenarioConfig(
        name="Test Scenario",
        default_curve=DefaultCurveConfig(
            vector=[0.01, 0.015, 0.02, 0.025, 0.02, 0.015, 0.01]
        ),
        prepayment_curve=PrepaymentCurveConfig(
            vector=[0.05, 0.07, 0.09, 0.11, 0.10, 0.09, 0.08]
        )
    )
    
    # Create a unique deal name for each test run to avoid cache conflicts
    unique_id = str(uuid.uuid4())[:8]
    
    return StructuredDealRequest(
        deal_name=f"Test Deal {unique_id}",
        pool=pool,
        waterfall=waterfall,
        scenario=scenario
    )

@pytest.fixture
def identical_deals():
    """Fixture to create two identical deals with the same name"""
    # Create a deal
    loan = LoanConfig(
        balance=200000.0,
        rate=0.045,
        term=360,
        start_date="2023-01-01",
        rate_type="fixed",
        payment_frequency="Monthly"
    )
    
    pool = LoanPoolConfig(
        pool_name="Cache Test Pool",
        loans=[loan]
    )
    
    waterfall = WaterfallConfig(
        start_date="2023-02-01",
        accounts=[
            AccountConfig(
                name="ReserveFund",
                initial_balance=5000.0
            )
        ],
        bonds=[
            BondConfig(
                name="Bond",
                balance=200000.0,
                rate=0.04
            )
        ],
        actions=[
            WaterfallAction(
                source="CollectedInterest",
                target="Bond",
                amount="Interest"
            ),
            WaterfallAction(
                source="CollectedPrincipal",
                target="Bond",
                amount="OutstandingPrincipal"
            )
        ]
    )
    
    scenario = ScenarioConfig(
        name="Cache Test Scenario",
        default_curve=DefaultCurveConfig(
            vector=[0.01, 0.01, 0.01]
        ),
        prepayment_curve=PrepaymentCurveConfig(
            vector=[0.05, 0.05, 0.05]
        )
    )
    
    # Use a fixed deal name so both deals are identical
    deal_name = "Cache Test Deal"
    
    # Create two identical deals
    deal1 = StructuredDealRequest(
        deal_name=deal_name,
        pool=pool,
        waterfall=waterfall,
        scenario=scenario
    )
    
    deal2 = StructuredDealRequest(
        deal_name=deal_name,
        pool=pool,
        waterfall=waterfall,
        scenario=scenario
    )
    
    return deal1, deal2

def test_service_initialization(absbox_service):
    """Test that the AbsBox service initializes correctly with cache"""
    assert absbox_service is not None
    assert absbox_service.use_cache is True
    assert absbox_service.redis is not None

def test_health_check(absbox_service):
    """Test the health check functionality"""
    health = absbox_service.health_check()
    
    # Verify health check response
    assert health is not None
    assert "status" in health
    assert health["status"] in ["healthy", "degraded"]
    
    # Verify cache status
    assert "cache" in health
    assert health["cache"]["status"] in ["connected", "error", "disabled"]
    
    # Verify engine status
    assert "engine" in health
    assert "status" in health["engine"]
    assert health["engine"]["status"] in ["connected", "error", "mock"]

def test_analyze_deal_success(absbox_service, sample_deal):
    """Test that analyzing a deal works and returns a success response"""
    response = absbox_service.analyze_deal(sample_deal)
    
    # Verify response
    assert response is not None
    assert response.status == "success"
    assert response.deal_name == sample_deal.deal_name
    assert hasattr(response, "execution_time")
    assert response.execution_time > 0
    
    # Verify cashflows
    assert hasattr(response, "bond_cashflows")
    assert hasattr(response, "pool_cashflows")
    assert len(response.bond_cashflows) > 0
    assert len(response.pool_cashflows) > 0

def test_cache_hit(absbox_service, redis_client, identical_deals):
    """Test that identical deals use the cache for the second request"""
    deal1, deal2 = identical_deals
    
    # Run first analysis and measure time
    start_time1 = time.time()
    response1 = absbox_service.analyze_deal(deal1)
    execution_time1 = time.time() - start_time1
    
    # Check that result was stored in cache
    cache_key = absbox_service._get_cache_key(deal1)
    assert redis_client.exists(cache_key) == 1
    
    # Run second analysis with identical deal and measure time
    start_time2 = time.time()
    response2 = absbox_service.analyze_deal(deal2)
    execution_time2 = time.time() - start_time2
    
    # Verify responses are equivalent
    assert response1.status == response2.status
    assert response1.deal_name == response2.deal_name
    
    # The second execution should generally be faster due to cache hit
    # But this is not guaranteed in a test environment, so we don't assert it
    
    # Verify cache metrics
    metrics = absbox_service.get_metrics()
    if metrics.get("enabled", False):
        cache_data = metrics.get("data", {})
        cache_hits = cache_data.get("absbox_cache_operations_total", {})
        assert any("operation=get,status=hit" in key for key in cache_hits.keys())

def test_cache_invalidation(absbox_service, redis_client, sample_deal):
    """Test that cache can be cleared"""
    # Run analysis to populate cache
    response1 = absbox_service.analyze_deal(sample_deal)
    assert response1.status == "success"
    
    # Verify cache contains data
    cache_keys = redis_client.keys("absbox:*")
    assert len(cache_keys) > 0
    
    # Clear cache
    clear_result = absbox_service.clear_cache()
    assert clear_result["status"] == "success"
    assert clear_result["keys_deleted"] > 0
    
    # Verify cache is empty
    cache_keys = redis_client.keys("absbox:*")
    assert len(cache_keys) == 0

def test_cache_ttl(absbox_service, redis_client, identical_deals):
    """Test that cache respects TTL settings"""
    deal1, deal2 = identical_deals
    
    # Set a very short TTL for this test
    absbox_service.cache_ttl = 1  # 1 second
    
    # Run first analysis
    response1 = absbox_service.analyze_deal(deal1)
    assert response1.status == "success"
    
    # Verify key exists with TTL
    cache_key = absbox_service._get_cache_key(deal1)
    assert redis_client.exists(cache_key) == 1
    assert redis_client.ttl(cache_key) <= absbox_service.cache_ttl
    
    # Wait for key to expire
    time.sleep(2)
    
    # Verify key no longer exists
    assert redis_client.exists(cache_key) == 0
    
    # Run second analysis (should be cache miss)
    response2 = absbox_service.analyze_deal(deal2)
    assert response2.status == "success"
    
    # The execution time should have been recalculated
    assert hasattr(response2, "execution_time")

def test_error_handling(absbox_service):
    """Test error handling in the service"""
    # Create an invalid deal (missing required fields)
    invalid_deal = StructuredDealRequest(
        deal_name="Invalid Deal",
        pool=None,  # Missing pool
        waterfall=None,  # Missing waterfall
        scenario=None   # Missing scenario
    )
    
    # Analyze should handle the error gracefully
    response = absbox_service.analyze_deal(invalid_deal)
    
    # Verify error response
    assert response is not None
    assert response.status == "error"
    assert response.deal_name == "Invalid Deal"
    assert hasattr(response, "error")
    assert response.error is not None
    assert hasattr(response, "error_type")
    assert response.error_type is not None

def test_metrics_collection(absbox_service, sample_deal):
    """Test that metrics are collected properly"""
    # Skip if metrics are not enabled
    if not absbox_service.get_metrics().get("enabled", False):
        pytest.skip("Metrics collection not enabled")
    
    # Run multiple analyses
    for _ in range(3):
        response = absbox_service.analyze_deal(sample_deal)
        assert response.status == "success"
    
    # Get metrics
    metrics = absbox_service.get_metrics()
    
    # Verify metrics structure
    assert "data" in metrics
    assert "timestamp" in metrics
    
    # Check for request counters
    data = metrics["data"]
    assert "absbox_request_count" in data
    assert "absbox_request_latency" in data
    
    # Verify request counts
    request_counts = data["absbox_request_count"]
    assert any("method=analyze_deal,status=success" in key for key in request_counts.keys())
    
    # Check that we have at least 3 successful requests
    success_keys = [k for k in request_counts.keys() if "method=analyze_deal,status=success" in k]
    if success_keys:
        assert request_counts[success_keys[0]] >= 3

if __name__ == "__main__":
    pytest.main(["-v", __file__])

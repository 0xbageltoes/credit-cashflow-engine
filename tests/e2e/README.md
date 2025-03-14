# End-to-End Tests for Monte Carlo Scenario Service

This directory contains comprehensive end-to-end tests for the Monte Carlo simulation service with scenario support. These tests use real backend services without mocks to ensure production readiness and resilience.

## Test Files Overview

### `test_monte_carlo_scenarios_api.py`

Tests the API endpoints for the Monte Carlo service, including:
- Creating, retrieving, updating, and deleting scenarios
- Running simulations with scenarios, both synchronous and asynchronous
- Comparing multiple scenarios
- Error handling and validation
- Cache behavior

### `test_frontend_client_integration.py`

Simulates how the frontend application would interact with the API, including:
- Client authentication patterns
- CRUD operations on scenarios
- Running simulations with progress polling
- Error handling patterns
- Scenario comparison and visualization data structures

### `test_redis_resilience.py`

Tests the resilience of the Redis caching system, simulating various failure modes:
- Connection failures
- Timeouts
- Data corruption
- Intermittent failures
- Large data handling

## Running the Tests

### Prerequisites

1. Make sure your `.env.test` file is properly configured with:
   - Valid Supabase credentials
   - Valid Upstash Redis credentials
   - All other required environment variables

2. Make sure the services are running:
   - Redis service must be accessible
   - Supabase should be accessible
   - API server should be running for API tests

### Run All E2E Tests

```bash
cd credit-cashflow-engine
pytest tests/e2e/ -v
```

### Run Specific Test Files

```bash
# Test API endpoints
pytest tests/e2e/test_monte_carlo_scenarios_api.py -v

# Test frontend client integration
pytest tests/e2e/test_frontend_client_integration.py -v

# Test Redis resilience
pytest tests/e2e/test_redis_resilience.py -v
```

### Run Specific Test Cases

```bash
# Example: Run only the Redis connection failure test
pytest tests/e2e/test_redis_resilience.py::test_connection_failure_resilience -v
```

## Important Notes

1. **Real Services**: These tests use real services to verify production behavior. 
   No mocks are used to ensure the tests represent actual production scenarios.

2. **Cleanup**: The tests include cleanup code to remove test data, but if tests 
   fail unexpectedly, some test data might remain in your Supabase database or Redis cache.

3. **Performance**: These tests are more time-consuming than unit tests since they 
   perform actual operations against real services.

4. **Error Handling**: The tests include comprehensive error handling patterns that 
   match how the production code handles failures.

5. **Redis Resilience**: The Redis resilience tests specifically validate that the 
   system continues to operate even when Redis is unavailable or misbehaving, ensuring 
   graceful degradation.

## Troubleshooting

### Redis Connection Issues

If you encounter Redis connection issues:

1. Verify your Redis environment variables in `.env.test`
2. Ensure Redis is running and accessible from your test environment
3. Check Redis connection timeouts and retry settings

### Supabase Connection Issues

If you encounter Supabase connection issues:

1. Verify your Supabase credentials in `.env.test`
2. Check if the Supabase service is accessible
3. Ensure your test user has appropriate permissions

### Slow Tests

If the tests are running too slowly:

1. Reduce `num_simulations` in the test fixtures
2. Run specific test cases instead of all tests
3. Consider running the tests in parallel with pytest-xdist

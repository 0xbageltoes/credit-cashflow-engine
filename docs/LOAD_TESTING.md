# Load Testing Guide for Credit Cashflow Engine

This guide provides instructions for load testing the Credit Cashflow Engine microservice to ensure it can handle expected production traffic and identify performance bottlenecks.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Running Load Tests](#running-load-tests)
4. [Test Scenarios](#test-scenarios)
5. [Interpreting Results](#interpreting-results)
6. [Performance Tuning](#performance-tuning)
7. [Continuous Load Testing](#continuous-load-testing)

## Introduction

Load testing is a critical step in ensuring the Credit Cashflow Engine can handle production workloads. The included Locust file simulates realistic user behavior and API usage patterns, allowing you to:

- Measure maximum throughput (requests per second)
- Identify performance bottlenecks
- Test API resilience under load
- Validate scaling configurations
- Establish baseline performance metrics

## Setup

### Prerequisites

- Python 3.8+
- Locust installed (`pip install locust`)
- A running instance of the Credit Cashflow Engine API
- (Optional) Grafana dashboard for monitoring during tests

### Installing Locust

```bash
pip install locust
```

### Configuration

Before running load tests, update the configuration in `locustfile.py`:

1. Set appropriate API authentication token:
   ```python
   TEST_CONFIG = {
       "use_fixed_token": True,
       "fixed_token": "YOUR_JWT_TOKEN_HERE",  # Update this
       # other config...
   }
   ```

2. Configure the test parameters:
   ```python
   TEST_CONFIG = {
       # ...
       "api_version": "v1",  # API version prefix
       "user_count": 10,     # Number of simulated users
       "ramp_up_time": 30,   # Seconds to ramp up load
       "run_time": 300,      # Total test duration in seconds
       # ...
   }
   ```

## Running Load Tests

### Basic Test

To run a basic load test against a local development server:

```bash
locust -f locustfile.py --host=http://localhost:8000
```

Then, access the Locust web UI at http://localhost:8089 to start the test.

### Headless Mode

For automated testing or CI/CD integration, run Locust in headless mode:

```bash
locust -f locustfile.py --host=http://api.example.com --users 50 --spawn-rate 10 --run-time 5m --headless --csv=results
```

### Advanced Options

#### Running Specific User Types

```bash
# Run only the CashflowUser class
locust -f locustfile.py --host=http://localhost:8000 --class-picker

# Run only specific tag groups
locust -f locustfile.py --host=http://localhost:8000 --tags forecast,scenarios
```

#### Distributed Load Testing

For generating higher load, distribute testing across multiple machines:

1. Start the master:
   ```bash
   locust -f locustfile.py --master --expect-workers=4
   ```

2. Start the workers (on other machines):
   ```bash
   locust -f locustfile.py --worker --master-host=<master_ip>
   ```

## Test Scenarios

The `locustfile.py` includes several test scenarios that simulate different user behaviors:

### Standard User Behavior

Simulates typical API usage, including:
- Health checks
- Standard cash flow forecasts
- Scenario management (create, load, list)
- History retrieval

### High-Intensity Scenarios

Simulates resource-intensive operations:
- Monte Carlo simulations
- Complex scenario analysis with many loans
- Sensitivity analysis

### Error Case Testing

Tests the API's error handling capabilities:
- Invalid input data
- Missing required fields
- Out-of-range values

## Interpreting Results

### Key Metrics to Monitor

- **Response Time**: The average, median, and 95th percentile response times
- **Requests Per Second (RPS)**: Maximum sustainable throughput
- **Error Rate**: Percentage of failed requests
- **System Metrics**: CPU, memory, and database utilization during the test

### Resource Utilization

During tests, monitor system resources:

1. API Server:
   - CPU usage
   - Memory consumption
   - Thread pool utilization

2. Database:
   - Connection pool utilization
   - Query performance
   - Lock contention

3. Redis Cache:
   - Hit rate
   - Memory usage
   - Eviction rate

## Performance Tuning

Based on load testing results, consider these common improvements:

### API Service Tuning

- Adjust worker count (`WORKERS` env var)
- Increase thread pool size (`CALCULATION_THREAD_POOL_SIZE` env var)
- Optimize calculation algorithms

### Caching Improvements

- Increase cache TTL for frequently accessed, rarely changed data
- Add additional cache layers for computation results
- Use Redis pipeline operations for bulk operations

### Database Optimizations

- Add indexes for frequently queried fields
- Optimize complex queries
- Consider read replicas for heavy read workloads

### Horizontal Scaling

- Increase the number of API instances
- Implement load balancing
- Consider sharding for very large datasets

## Continuous Load Testing

Integrate load testing into your CI/CD pipeline:

1. Create a load test job that runs against staging environments after deployment
2. Set performance thresholds for pass/fail criteria
3. Maintain a performance benchmark database to track changes over time

### Example CI Integration

```yaml
load_test:
  stage: test
  script:
    - pip install locust
    - locust -f locustfile.py --host=https://staging-api.example.com --users 30 --spawn-rate 10 --run-time 3m --headless --csv=results
    - python scripts/analyze_load_test_results.py --csv results --threshold-rps 50 --threshold-error-rate 0.01
  artifacts:
    paths:
      - results*.csv
    expire_in: 1 week
```

## Best Practices

- **Test incrementally**: Start with low load and gradually increase
- **Test in isolation**: Ensure other processes don't interfere with results
- **Test regularly**: Schedule regular load tests, not just before releases
- **Monitor everything**: Collect as much telemetry as possible during tests
- **Document findings**: Keep a record of issues found and improvements made

---

For more information or assistance with load testing, contact the development team at cashflow-engine-support@example.com.

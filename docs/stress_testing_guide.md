# Asset Classes Stress Testing Guide

This document provides comprehensive documentation for the stress testing module in the credit cashflow engine, including setup instructions, API usage, and operational guidance for production environments.

## Overview

The stress testing module allows financial analysts to apply various stress scenarios to asset pools to evaluate their performance under different market conditions. This helps identify potential risks and understand the sensitivity of asset valuations to economic changes.

Key capabilities:
- Pre-defined industry-standard stress scenarios
- Custom scenario definition
- Parallel processing for multiple scenarios
- Redis caching integration for performance optimization
- Comprehensive reporting and metrics
- WebSocket support for real-time updates on long-running tests

## API Endpoints

### Run Stress Tests

```
POST /api/v1/asset-classes/stress-testing/run
```

Runs a series of stress tests on an asset pool and returns detailed results for each scenario.

**Request Body:**
```json
{
  "request": {
    "pool": {
      "pool_id": "string",
      "pool_name": "string",
      "assets": [
        {
          "asset_id": "string",
          "asset_class": "residential_mortgage",
          "balance": 250000,
          "rate": 0.0375,
          "term": 360,
          "age": 12,
          "original_balance": 275000,
          "rate_type": "fixed",
          "prepayment_speed": 0.07,
          "default_probability": 0.02,
          "recovery_rate": 0.65,
          "loss_severity": 0.35,
          "status": "performing"
        }
      ],
      "cutoff_date": "2023-10-01",
      "metadata": {
        "source": "string",
        "description": "string"
      }
    },
    "analysis_date": "2023-10-01",
    "discount_rate": 0.05,
    "projection_periods": 60,
    "include_cashflows": false
  },
  "scenario_names": ["base", "rate_shock_up", "credit_crisis"],
  "custom_scenarios": {
    "custom_scenario": {
      "name": "Custom Scenario",
      "description": "My custom stress scenario",
      "market_factors": {
        "interest_rate_shock": 0.04,
        "default_multiplier": 2.5,
        "recovery_multiplier": 0.7
      }
    }
  },
  "run_parallel": true,
  "max_workers": 4,
  "include_cashflows": false,
  "generate_report": true
}
```

**Query Parameters:**
- `use_cache` (boolean, optional): Whether to use Redis caching for results. Default: true

**Response:**
A JSON object mapping scenario names to detailed analysis results:

```json
{
  "base": {
    "pool_name": "string",
    "analysis_date": "2023-10-01",
    "status": "success",
    "execution_time": 1.25,
    "metrics": {
      "npv": 273621.45,
      "irr": 0.0612,
      "total_principal": 285000.0,
      "total_interest": 42318.55,
      "duration": 3.2,
      "weighted_average_life": 4.7,
      "npv_change": 0,
      "npv_change_percent": 0
    },
    "analytics": {
      "stress_test_report": {
        "pool_name": "string",
        "analysis_date": "2023-10-01",
        "execution_time_total": 5.75,
        "scenarios_count": 3,
        "scenarios_success": 3,
        "scenarios_failed": 0,
        "base_case": {
          "npv": 273621.45,
          "total_principal": 285000.0,
          "total_interest": 42318.55,
          "duration": 3.2,
          "weighted_average_life": 4.7
        },
        "scenarios": {
          "rate_shock_up": {
            "name": "rate_shock_up",
            "status": "success",
            "execution_time": 1.25,
            "cache_hit": false,
            "metrics": {
              "npv": 260000.76,
              "npv_change": -13620.69,
              "npv_change_percent": -4.98,
              "duration": 2.9,
              "weighted_average_life": 4.3
            }
          },
          "credit_crisis": {
            "name": "credit_crisis",
            "status": "success",
            "execution_time": 1.5,
            "cache_hit": false,
            "metrics": {
              "npv": 225000.12,
              "npv_change": -48621.33,
              "npv_change_percent": -17.77,
              "duration": 3.0,
              "weighted_average_life": 4.5
            }
          }
        }
      }
    },
    "cache_hit": false
  },
  "rate_shock_up": {
    "pool_name": "string",
    "analysis_date": "2023-10-01",
    "status": "success",
    "execution_time": 1.25,
    "metrics": {
      "npv": 260000.76,
      "irr": 0.0582,
      "total_principal": 285000.0,
      "total_interest": 39000.55,
      "duration": 2.9,
      "weighted_average_life": 4.3,
      "npv_change": -13620.69,
      "npv_change_percent": -4.98
    },
    "analytics": {
      "scenario": {
        "name": "Rate Shock Up",
        "description": "Interest rates increase by 300 basis points",
        "market_factors": {
          "interest_rate_shock": 0.03,
          "prepayment_multiplier": 0.7,
          "default_multiplier": 1.2
        }
      }
    },
    "cache_hit": false
  },
  "credit_crisis": {
    "pool_name": "string",
    "analysis_date": "2023-10-01",
    "status": "success",
    "execution_time": 1.5,
    "metrics": {
      "npv": 225000.12,
      "irr": 0.0512,
      "total_principal": 265000.0,
      "total_interest": 38000.55,
      "duration": 3.0,
      "weighted_average_life": 4.5,
      "npv_change": -48621.33,
      "npv_change_percent": -17.77
    },
    "analytics": {
      "scenario": {
        "name": "Credit Crisis",
        "description": "Severe economic downturn with credit deterioration",
        "market_factors": {
          "default_multiplier": 3.0,
          "recovery_multiplier": 0.6,
          "prepayment_multiplier": 0.5,
          "interest_rate_shock": 0.01
        }
      }
    },
    "cache_hit": false
  }
}
```

### Run Stress Tests Asynchronously

```
POST /api/v1/asset-classes/stress-testing/async-run
```

Queues stress tests to run asynchronously in the background and provides real-time updates via WebSocket.

**Request Body:**
Same as the synchronous endpoint.

**Query Parameters:**
- `use_cache` (boolean, optional): Whether to use Redis caching for results. Default: true

**Response:**
```json
{
  "task_id": "12345678-1234-5678-1234-567812345678",
  "status": "queued",
  "message": "Stress tests queued successfully",
  "pool_name": "Sample Pool",
  "scenario_count": 3,
  "websocket_url": "/ws/user123?task_id=12345678-1234-5678-1234-567812345678"
}
```

### Get Available Stress Scenarios

```
GET /api/v1/asset-classes/stress-testing/scenarios
```

Returns a list of predefined stress test scenarios available in the system.

**Response:**
```json
{
  "base": {
    "name": "Base Case",
    "description": "Standard market conditions",
    "market_factors": {}
  },
  "rate_shock_up": {
    "name": "Rate Shock Up",
    "description": "Interest rates increase by 300 basis points",
    "market_factors": {
      "interest_rate_shock": 0.03,
      "prepayment_multiplier": 0.7,
      "default_multiplier": 1.2
    }
  },
  "credit_crisis": {
    "name": "Credit Crisis",
    "description": "Severe economic downturn with credit deterioration",
    "market_factors": {
      "default_multiplier": 3.0,
      "recovery_multiplier": 0.6,
      "prepayment_multiplier": 0.5,
      "interest_rate_shock": 0.01
    }
  }
}
```

## WebSocket Integration

The stress testing module supports real-time updates for long-running stress tests via WebSocket. 

### Connection

Connect to the WebSocket using the URL returned by the async-run endpoint:

```
ws://<server-host>/ws/<user_id>?task_id=<task_id>
```

### Message Format

The server will send JSON messages with the following structure:

```json
{
  "task_id": "12345678-1234-5678-1234-567812345678",
  "status": "running",
  "message": "Running base case analysis",
  "data": {
    "progress": 10,
    "current_scenario": "base"
  }
}
```

Status values:
- `queued`: The task is in the queue waiting to be executed
- `running`: The task is being executed
- `completed`: The task has completed successfully
- `error`: An error occurred during task execution

When status is `completed`, the `data` will contain the full results.

## Redis Cache Configuration

The stress testing module integrates with Redis for caching results. This significantly improves performance for repeated analyses.

### Configuration Parameters

The following environment variables control Redis behavior:

| Variable | Description | Default |
|----------|-------------|---------|
| REDIS_HOST | Redis server hostname | localhost |
| REDIS_PORT | Redis server port | 6379 |
| REDIS_PASSWORD | Redis authentication password | None |
| REDIS_DB | Redis database number | 0 |
| REDIS_SSL | Whether to use SSL/TLS | False |
| REDIS_SOCKET_TIMEOUT | Socket timeout in seconds | 5 |
| REDIS_CONNECT_TIMEOUT | Connection timeout in seconds | 1 |
| REDIS_MAX_CONNECTIONS | Maximum connections in the pool | 10 |
| STRESS_TEST_CACHE_TTL | Time-to-live for cached results (seconds) | 3600 |

### Cache Key Strategy

Cache keys for stress test results use the following format:
```
stress_test_user_{user_id}_pool_{pool_name}_scenario_{scenario_name}_date_{analysis_date}_discount_{discount_rate}
```

This ensures deterministic caching while accounting for key parameters that affect results.

## Predefined Stress Scenarios

The module includes the following predefined stress scenarios:

| Scenario | Description | Market Factors |
|----------|-------------|---------------|
| Base Case | Standard market conditions | N/A |
| Rate Shock Up | Interest rates increase by 300 basis points | interest_rate_shock: +0.03, prepayment_multiplier: 0.7, default_multiplier: 1.2 |
| Rate Shock Down | Interest rates decrease by 200 basis points | interest_rate_shock: -0.02, prepayment_multiplier: 1.5, default_multiplier: 0.9 |
| Credit Crisis | Severe economic downturn with credit deterioration | default_multiplier: 3.0, recovery_multiplier: 0.6, prepayment_multiplier: 0.5, interest_rate_shock: +0.01 |
| Liquidity Crisis | Market-wide liquidity constraints | spread_widening: 0.05, prepayment_multiplier: 0.4, default_multiplier: 1.8 |
| Housing Boom | Rapid appreciation in property values | prepayment_multiplier: 2.0, recovery_multiplier: 1.3, default_multiplier: 0.7 |
| Housing Bust | Rapid depreciation in property values | recovery_multiplier: 0.5, default_multiplier: 2.5, prepayment_multiplier: 0.6 |

## Custom Scenario Definition

Custom scenarios can be defined with the following parameters:

| Parameter | Description | Example |
|-----------|-------------|---------|
| name | Descriptive name | "Custom Inflation Scenario" |
| description | Detailed description | "High inflation environment with elevated interest rates" |
| market_factors | Key-value pairs of stress factors | See below |

### Available Market Factors

| Factor | Description | Value Range |
|--------|-------------|------------|
| interest_rate_shock | Change in interest rates (absolute) | -0.05 to +0.10 |
| discount_rate_shock | Change in discount rate (absolute) | -0.05 to +0.10 |
| default_multiplier | Multiplier for default probabilities | 0.5 to 5.0 |
| recovery_multiplier | Multiplier for recovery rates | 0.2 to 1.5 |
| prepayment_multiplier | Multiplier for prepayment rates | 0.2 to 3.0 |
| spread_widening | Credit spread widening (absolute) | 0.0 to 0.20 |

## Production Deployment

### Resource Requirements

For optimal performance in a production environment:

- **CPU**: At least 4 cores recommended for parallel processing
- **Memory**: Minimum 8GB RAM, 16GB recommended for larger pools
- **Redis**: Dedicated Redis instance with at least 1GB memory
- **Disk**: SSD storage recommended for optimal I/O performance

### Scaling Strategies

1. **Horizontal Scaling**:
   - Deploy multiple API instances behind a load balancer
   - Use a centralized Redis cache for shared caching

2. **Vertical Scaling**:
   - Increase available CPU cores to improve parallel processing
   - Increase memory for larger asset pools

### Monitoring

Key metrics to monitor:

1. **Performance Metrics**:
   - Execution time per scenario
   - Total stress test execution time
   - Cache hit rates

2. **Error Rates**:
   - API endpoint errors
   - Internal calculation errors
   - Redis connection issues

3. **Resource Utilization**:
   - CPU usage during parallel runs
   - Memory consumption with large asset pools
   - Redis memory usage and eviction rates

### Logging

The module uses structured logging with the following levels:

- **INFO**: Standard operation information
- **WARNING**: Issues that might require attention but don't affect results
- **ERROR**: Issues that affect results or prevent stress tests from completing
- **DEBUG**: Detailed information for troubleshooting

### Health Checks

Implement health checks for:

1. API endpoints:
   ```
   GET /api/v1/health/stress-testing
   ```

2. Redis connectivity:
   ```
   GET /api/v1/health/redis
   ```

### Rate Limiting

Implement rate limiting for API endpoints to prevent resource exhaustion:

```
# Example with nginx
limit_req_zone $binary_remote_addr zone=stress_testing:10m rate=5r/m;

location /api/v1/asset-classes/stress-testing/ {
    limit_req zone=stress_testing burst=10 nodelay;
    proxy_pass http://backend;
}
```

## Error Handling and Troubleshooting

### Common Errors

| Error | Possible Causes | Resolution |
|-------|----------------|------------|
| 400 Bad Request | Invalid request format or empty asset pool | Verify request structure and ensure asset pool contains assets |
| 401 Unauthorized | Missing or invalid authentication token | Check authentication credentials |
| 500 Internal Server Error | Server-side calculation error | Check logs for detailed error information |
| Redis Connection Error | Redis server unavailable | Verify Redis connection settings |

### Troubleshooting Steps

1. **API Errors**:
   - Check request payload against API documentation
   - Review server logs for detailed error messages

2. **Performance Issues**:
   - Reduce pool size or number of scenarios
   - Check Redis cache configuration
   - Monitor system resources during execution

3. **Redis Issues**:
   - Verify Redis server is running
   - Check network connectivity to Redis
   - Review Redis configuration

## Best Practices

### Optimal Usage

1. **Asset Pool Size**:
   - Limit pools to 1,000 assets for synchronous calls
   - Use async endpoint for larger pools

2. **Scenario Selection**:
   - Run only necessary scenarios to reduce processing time
   - Use the base case and 2-3 relevant stress scenarios

3. **Caching Strategy**:
   - Enable caching for repeated analyses
   - Configure appropriate TTL based on data update frequency

### Data Preparation

1. **Asset Data Quality**:
   - Ensure all required fields are present
   - Validate rate types and numeric values
   - Use consistent units for monetary values

2. **Pool Homogeneity**:
   - Group similar assets in pools for better performance
   - Consider creating separate pools by asset class

## Appendix

### Sample Code

#### Python Client Example

```python
import requests
import json
import websocket
import threading
import time

# Synchronous call
def run_stress_test(api_url, token, request_data):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{api_url}/api/v1/asset-classes/stress-testing/run",
        headers=headers,
        json=request_data
    )
    
    return response.json()

# Asynchronous call with WebSocket updates
def run_async_stress_test(api_url, ws_url, token, request_data):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Start async task
    response = requests.post(
        f"{api_url}/api/v1/asset-classes/stress-testing/async-run",
        headers=headers,
        json=request_data
    )
    
    task_info = response.json()
    task_id = task_info["task_id"]
    ws_endpoint = task_info["websocket_url"]
    
    results = {"completed": False, "data": None}
    
    # WebSocket handler
    def on_message(ws, message):
        data = json.loads(message)
        print(f"Status: {data['status']} - {data['message']}")
        
        if data["status"] == "completed":
            results["completed"] = True
            results["data"] = data["data"]["results"]
            ws.close()
    
    # Connect to WebSocket for updates
    ws = websocket.WebSocketApp(
        f"{ws_url}{ws_endpoint}",
        on_message=on_message
    )
    
    # Start WebSocket connection in a separate thread
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    
    # Wait for completion or timeout
    timeout = 300  # 5 minutes
    start_time = time.time()
    
    while not results["completed"] and (time.time() - start_time) < timeout:
        time.sleep(1)
    
    return results["data"]
```

### Glossary

| Term | Definition |
|------|------------|
| NPV | Net Present Value - the discounted value of future cash flows |
| IRR | Internal Rate of Return - the interest rate at which NPV equals zero |
| Duration | Weighted average time until cash flows are received |
| WAL | Weighted Average Life - average time until principal is repaid |
| Stress Test | Analysis of asset performance under adverse conditions |
| Market Factors | Parameters that modify asset behavior in stress scenarios |

# AbsBox Configuration Guide

This guide provides detailed information on configuring and deploying the AbsBox integration across different environments (development, testing, and production).

## Table of Contents

1. [Overview](#overview)
2. [Environment Variables](#environment-variables)
3. [Development Environment](#development-environment)
4. [Testing Environment](#testing-environment)
5. [Production Environment](#production-environment)
6. [Hastructure Engine Configuration](#hastructure-engine-configuration)
7. [Redis Cache Configuration](#redis-cache-configuration)
8. [Performance Tuning](#performance-tuning)
9. [Monitoring and Metrics](#monitoring-and-metrics)
10. [Troubleshooting](#troubleshooting)

## Overview

The AbsBox integration allows the credit cashflow engine to perform structured finance analysis using the AbsBox library and Hastructure computation engine. The integration is designed to be flexible and can be configured to work in various environments with different setups.

### Key Components

- **AbsBox Library**: Python library for structured finance analysis
- **Hastructure Engine**: Computation engine for running AbsBox models
- **Redis Cache**: Optional cache for storing calculation results
- **Monitoring Dashboard**: Optional dashboard for monitoring performance metrics

## Environment Variables

The following environment variables control the AbsBox integration:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HASTRUCTURE_ENGINE_URL` | URL of the Hastructure engine | `http://localhost:5000` | Yes |
| `HASTRUCTURE_MAX_POOL_SIZE` | Maximum thread pool size for calculations | `4` | No |
| `HASTRUCTURE_CALCULATION_TIMEOUT` | Timeout in seconds for calculations | `300` | No |
| `USE_MOCK_HASTRUCTURE` | Use mock engine instead of real one | `false` | No |
| `ABSBOX_DASHBOARD_PORT` | Port for the monitoring dashboard | `5000` | No |
| `ABSBOX_CACHE_ENABLED` | Enable Redis caching for AbsBox | `true` | No |
| `ABSBOX_CACHE_TTL` | Time-to-live in seconds for cached items | `3600` | No |
| `ENVIRONMENT` | Current environment (`development`, `test`, `production`) | `development` | Yes |
| `REDIS_URL` | URL of the Redis server |  | No |

## Development Environment

In the development environment, you can run a local Hastructure engine or use the mock engine provided by AbsBox.

### Using the Mock Engine

For quick development without setting up the full Hastructure engine:

1. Set `USE_MOCK_HASTRUCTURE=true` in your `.env.development` file
2. Install AbsBox with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

The mock engine provides simulated responses that are useful for testing your implementation without requiring the full engine setup.

### Using a Local Hastructure Engine

For a more realistic development experience:

1. Set `HASTRUCTURE_ENGINE_URL=http://localhost:5000` in your `.env.development` file
2. Run the setup script to configure a local Hastructure instance:
   ```bash
   python setup_local_absbox.py
   ```
3. Start the local engine:
   ```bash
   python -m absbox.local_engine
   ```

## Testing Environment

For the testing environment, we recommend using a combination of real and mock components depending on the test:

1. Set the following in your `.env.test` file:
   ```
   ENVIRONMENT=test
   USE_MOCK_HASTRUCTURE=true
   ABSBOX_CACHE_ENABLED=true
   ```

2. For unit tests, use the mock engine:
   ```python
   # In your test file
   os.environ["USE_MOCK_HASTRUCTURE"] = "true"
   ```

3. For integration tests, you can use a real Hastructure engine inside a Docker container:
   ```bash
   # Start the container before running tests
   docker-compose -f docker-compose.test.yml up -d hastructure
   ```

## Production Environment

In production, you should use the full Hastructure engine running in a dedicated container:

1. Configure your `.env.production` file:
   ```
   ENVIRONMENT=production
   HASTRUCTURE_ENGINE_URL=http://hastructure:5000
   HASTRUCTURE_MAX_POOL_SIZE=8
   ABSBOX_CACHE_ENABLED=true
   ABSBOX_CACHE_TTL=3600
   ```

2. Ensure the Hastructure service is properly configured in your `docker-compose.prod.yml`:
   ```yaml
   hastructure:
     image: hastructure/engine:latest
     environment:
       - MAX_WORKERS=8
       - LOG_LEVEL=INFO
     ports:
       - "127.0.0.1:5000:5000"
     volumes:
       - hastructure_data:/data
     restart: unless-stopped
     healthcheck:
       test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
       interval: 30s
       timeout: 10s
       retries: 3
       start_period: 20s
   ```

3. Set up proper resource limits for the container:
   ```yaml
   hastructure:
     # ...other configs
     deploy:
       resources:
         limits:
           cpus: '4'
           memory: 8G
         reservations:
           cpus: '2'
           memory: 4G
   ```

## Hastructure Engine Configuration

The Hastructure engine has several configuration options that can be adjusted based on your needs:

### Container Configuration

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|------------|
| `MAX_WORKERS` | Number of worker processes | `4` | Match CPU cores |
| `THREAD_POOL_SIZE` | Threads per worker | `4` | 2-4 per core |
| `LOG_LEVEL` | Logging verbosity | `INFO` | `INFO` or `WARNING` in prod |
| `CALCULATION_TIMEOUT` | Maximum time for a calculation | `300` | 300-600 seconds |
| `MEMORY_LIMIT` | Maximum memory per calculation | `1G` | 2-4GB in production |

### Engine Performance Options

You can also adjust the calculation engine parameters:

```yaml
hastructure:
  # ...other configs
  environment:
    # Standard options
    - MAX_WORKERS=8
    - LOG_LEVEL=INFO
    
    # Advanced options
    - PRECISION=double    # Numerical precision (float/double)
    - CACHE_SIZE=512MB    # Internal calculation cache size
    - REPORT_INTERVAL=5    # Progress reporting interval in seconds
    - OPTIMIZATION_LEVEL=2    # Calculation optimization level (0-3)
```

## Redis Cache Configuration

The AbsBox service supports caching analysis results using Redis to improve performance and reduce load on the AbsBox API. This section describes how to configure and use the Redis caching feature.

### Prerequisites

- Redis server (v6.0 or newer recommended)
- Redis Python client library (`pip install redis`)

### Configuration

#### Environment Variables

Set the following environment variables to configure Redis:

```bash
# Required for Redis connection
REDIS_URL=redis://user:password@hostname:port/db

# Optional cache configuration
ABSBOX_CACHE_ENABLED=true  # Set to 'false' to disable caching
ABSBOX_CACHE_TTL=3600      # Time-to-live for cached items in seconds (default: 3600)
```

For local development, you might use:
```bash
REDIS_URL=redis://localhost:6379/0
```

For production, consider using a managed Redis service like Upstash, Redis Labs, or AWS ElastiCache.

#### Redis Configuration Parameters

The service uses the following default Redis client configuration parameters which can be customized in code:

| Parameter | Default | Description |
|-----------|---------|-------------|
| socket_timeout | 5.0 | Socket timeout in seconds |
| socket_connect_timeout | 5.0 | Connection timeout in seconds |
| retry_on_timeout | true | Whether to retry operations on timeout |
| health_check_interval | 30 | Health check interval in seconds |
| max_connections | 10 | Maximum number of connections in the connection pool |
| decode_responses | false | Whether to decode responses to strings |

### Usage

The AbsBoxServiceEnhanced class will automatically use Redis caching if:

1. The Redis URL is properly configured in the environment
2. The Redis Python client is installed and available
3. Caching is not explicitly disabled through config

You can explicitly configure caching by passing configuration parameters:

```python
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced

# Create service with custom cache configuration
service = AbsBoxServiceEnhanced({
    "use_cache": True,
    "cache_ttl": 7200  # 2 hours cache TTL
})
```

### Troubleshooting

If Redis connection fails, the service will log errors but continue to function without caching. Check the logs for error messages like:

```
ERROR:absbox_service:Failed to initialize Redis client: Error connecting to Redis at hostname:port
WARNING:absbox_service:Continuing without Redis cache
```

Common issues:
- Incorrect Redis URL format
- Network connectivity issues 
- Missing Redis client library
- Redis server not running or inaccessible

To manually verify Redis connectivity:

```python
import redis
r = redis.from_url("your-redis-url")
r.ping()  # Should return True if connection is successful
```

## Performance Tuning

To optimize the AbsBox integration performance:

### Caching Strategy

1. **TTL Optimization**: Set different TTLs based on calculation complexity:
   ```python
   if deal.is_complex():
     cache_ttl = 86400  # 24 hours for complex calculations
   else:
     cache_ttl = 3600   # 1 hour for simple calculations
   ```

2. **Precomputed Results**: For common scenarios, precompute and cache results during off-peak hours.

### Resource Allocation

1. **Thread Pool Sizing**: Set `HASTRUCTURE_MAX_POOL_SIZE` to match available CPU cores.
2. **Memory Management**: Monitor memory usage and adjust container limits accordingly.
3. **Timeout Management**: Set appropriate timeouts for calculations based on complexity.

## Monitoring and Metrics

The AbsBox integration includes a monitoring dashboard and Prometheus metrics:

### Dashboard Setup

1. Start the monitoring dashboard:
   ```bash
   python monitoring/absbox_monitoring.py
   ```

2. Access the dashboard at `http://localhost:5000` (or configured port)

### Metrics Collection

The AbsBox service exposes the following metrics:

- `absbox_request_count`: Number of requests by method and status
- `absbox_request_latency`: Request duration in seconds
- `absbox_error_count`: Count of errors by type
- `absbox_cache_hits`: Number of cache hits
- `absbox_cache_misses`: Number of cache misses
- `absbox_active_calculations`: Number of active calculations

### Prometheus Integration

To collect these metrics with Prometheus:

1. Add the metrics endpoint to your Prometheus configuration:
   ```yaml
   scrape_configs:
     - job_name: 'absbox'
       scrape_interval: 15s
       metrics_path: '/metrics'
       static_configs:
         - targets: ['absbox_service:5000']
   ```

2. Create Grafana dashboards to visualize the metrics.

## Troubleshooting

Common issues and their solutions:

### Engine Connection Issues

If you're having trouble connecting to the Hastructure engine:

1. Verify the engine is running:
   ```bash
   curl http://localhost:5000/health
   ```

2. Check network connectivity and firewall settings:
   ```bash
   telnet localhost 5000
   ```

3. Review logs for connection errors:
   ```bash
   docker logs hastructure
   ```

### Performance Problems

If calculations are slow:

1. Check system resources during calculation:
   ```bash
   docker stats hastructure
   ```

2. Enable debug logging temporarily:
   ```
   LOG_LEVEL=DEBUG
   ```

3. Review the calculation parameters for complexity.

### Cache Issues

If the cache isn't working correctly:

1. Verify Redis connection:
   ```bash
   redis-cli ping
   ```

2. Check cache hit rates in the monitoring dashboard.

3. Clear the cache if data becomes stale:
   ```python
   absbox_service.clear_cache()
   ```

### Mock Engine Issues

If the mock engine isn't functioning correctly:

1. Verify the mock flag is correctly set:
   ```bash
   echo $USE_MOCK_HASTRUCTURE
   ```

2. Reinstall AbsBox with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Check for AbsBox version compatibility issues.

---

For additional assistance with configuring or troubleshooting the AbsBox integration, please refer to the AbsBox documentation or contact the development team.

# Monte Carlo Simulation Scenario Management

## Overview

The Monte Carlo Simulation Scenario Management system provides a comprehensive suite of capabilities for creating, running, and comparing financial simulations with different economic scenarios. This documentation covers the technical details of the implementation, API endpoints, testing, and best practices for production use.

## Features

- **Full Scenario Management**: Create, update, delete, and retrieve scenario definitions
- **Scenario-Based Simulations**: Run Monte Carlo simulations with specific scenarios applied
- **Scenario Comparison**: Compare results across multiple scenarios
- **Robust Caching**: Redis-based caching for optimal performance with proper error handling
- **Resilient Architecture**: Graceful degradation when services are unavailable
- **Comprehensive Error Handling**: Detailed error logging and recovery mechanisms
- **Background Processing**: Support for asynchronous execution via Celery workers
- **API Integration**: RESTful API for all scenario management functions

## API Endpoints

### Scenario Management

- `POST /v1/monte-carlo/scenarios`: Create a new scenario definition
- `GET /v1/monte-carlo/scenarios`: List available scenarios
- `GET /v1/monte-carlo/scenarios/{scenario_id}`: Retrieve a specific scenario
- `PUT /v1/monte-carlo/scenarios/{scenario_id}`: Update a scenario
- `DELETE /v1/monte-carlo/scenarios/{scenario_id}`: Delete a scenario

### Simulation with Scenarios

- `POST /v1/monte-carlo/simulations/with-scenario`: Run a simulation with a specific scenario
- `POST /v1/monte-carlo/scenarios/compare`: Compare multiple scenarios

## Code Structure

### Core Components

- **MonteCarloSimulationService**: Main service for running simulations
- **SupabaseService**: Database service for storing/retrieving scenarios and simulation results
- **RedisService**: Caching service with robust error handling
- **Monte Carlo Celery Workers**: Background task processing for long-running simulations

### Key Methods

- `run_simulation_with_scenario`: Runs a simulation with a specific scenario applied
- `apply_scenario`: Applies scenario modifiers to a simulation request
- `compare_scenarios`: Runs and compares simulations across multiple scenarios

## Scenario Definition

A scenario is defined by:

```json
{
  "id": "unique-scenario-id",
  "name": "Scenario Name",
  "type": "scenario_type",
  "description": "Scenario description",
  "user_id": "owner_user_id",
  "parameters": {
    "risk_factor_modifiers": {
      "variable_name": {
        "mean_shift": 0.02,
        "volatility_multiplier": 1.5
      }
    },
    "correlation_modifiers": {
      "var1:var2": 0.1
    },
    "additional_parameters": {
      "custom_param1": "value"
    }
  }
}
```

## Best Practices

### Performance Optimization

1. **Caching Strategy**: 
   - Cache keys include scenario IDs for scenario-specific caching
   - TTL settings are configurable via environment variables
   - Cache invalidation occurs automatically when simulations or scenarios are updated

2. **Memory Management**:
   - Large simulation paths are stored separately to avoid memory issues
   - Simulations with many iterations are automatically processed in worker tasks

### Error Handling

1. **Redis Failures**:
   - All Redis operations are wrapped with proper error handling
   - System gracefully degrades to non-cached operation if Redis is unavailable
   - Detailed logging helps identify Redis connection issues

2. **Database Errors**:
   - Database operations include retry logic
   - Failed operations are logged with detailed error information
   - Temporary database unavailability won't crash the application

### Production Deployment

1. **Environment Configuration**:
   - Redis connection parameters are configurable via environment variables
   - Cache TTL and timeout settings are also configurable
   - Worker concurrency can be adjusted based on workload

2. **Monitoring**:
   - Simulation execution times are tracked and logged
   - Error rates for simulations are available for monitoring
   - Cache hit/miss rates can be monitored

## Testing

The system includes comprehensive unit tests covering:

1. Scenario application to simulation requests
2. Redis cache functionality with failure simulations
3. API endpoint integration tests
4. Error handling in various failure scenarios
5. Cache key generation and uniqueness

Run tests using:

```bash
pytest tests/test_monte_carlo_scenarios.py -v
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failures**:
   - Check Redis connection string in environment variables
   - Verify that Redis service is running and accessible
   - Look for timeout or connection refused errors in logs

2. **Slow Simulation Performance**:
   - Check Redis cache hit rates
   - Verify that appropriate caching is enabled
   - Consider increasing worker count for parallel processing

3. **Invalid Scenario Parameters**:
   - Ensure scenario parameters follow the required format
   - Check for unsupported modifiers for specific variable types
   - Verify that correlation modifiers keep values within [-1, 1]

### Logging

The system uses structured logging with different verbosity levels:

- **ERROR**: Critical issues that prevent simulation execution
- **WARNING**: Non-critical issues that might affect results
- **INFO**: Important operational information
- **DEBUG**: Detailed information for troubleshooting

## Related Documentation

- [Redis Cache Configuration](redis_configuration.md)
- [Supabase Integration](supabase_integration.md)
- [Monte Carlo Simulation Models](monte_carlo_models.md)
- [API Documentation](api_documentation.md)

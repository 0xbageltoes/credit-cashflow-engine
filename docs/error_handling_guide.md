# Error Handling Guide for Credit Cashflow Engine

## Overview

This guide documents the unified error handling framework implemented in the credit-cashflow-engine. The framework provides a consistent pattern for handling errors across the application, improving debuggability, observability, and user experience.

## Key Components

1. **Base Error Classes**: Hierarchy of error types for different parts of the application
2. **Error Handling Decorator**: Wraps functions to provide consistent error handling
3. **Exception Handlers**: Consistent API responses for application errors
4. **Helper Functions**: Utilities for working with errors

## Base Error Classes

All application errors inherit from `ApplicationError`, which captures:

- Error message
- Contextual information
- Original exception (cause)
- Timestamp
- Serialization to dictionary for logging

### Error Hierarchy

```
ApplicationError
├── CalculationError      # Financial calculation errors
├── DataError             # Data validation/parsing errors
├── ServiceError          # External service errors
├── ConfigurationError    # Configuration/setup errors
├── CacheError            # Caching operation errors
├── DatabaseError         # Database operation errors
└── ValidationError       # Input validation errors
```

## Using the Error Framework

### Creating Custom Errors

```python
# Import base classes
from app.core.error_handling import ApplicationError, CalculationError

# Raising a basic error
raise CalculationError("Failed to calculate NPV")

# With context
raise CalculationError(
    message="Failed to calculate NPV", 
    context={"loan_id": loan_id, "discount_rate": rate}
)

# With original exception
try:
    result = complex_calculation(data)
except ValueError as e:
    raise CalculationError(
        message="Calculation failed with invalid value",
        context={"data_point": data},
        cause=e
    )
```

### Using the Error Handling Decorator

Decorate functions to automatically handle and convert exceptions:

```python
from app.core.error_handling import handle_errors, CalculationError, DataError
import logging

logger = logging.getLogger(__name__)

# Error mapping for specific exception types
error_mapping = {
    ValueError: DataError,
    ZeroDivisionError: CalculationError
}

@handle_errors(logger=logger, error_mapping=error_mapping)
async def calculate_risk_metrics(loan_data):
    # Function implementation
    # Any exceptions will be converted to ApplicationErrors
    # with proper context and logging
    return risk_metrics
```

The decorator supports both synchronous and asynchronous functions automatically.

### Handling API Errors

FastAPI exception handlers are automatically registered for consistent API error responses:

```python
# In an API endpoint
from fastapi import APIRouter, Depends
from app.core.error_handling import ValidationError

router = APIRouter()

@router.post("/calculate")
async def calculate_endpoint(data: CalculationRequest):
    if data.amount <= 0:
        raise ValidationError(
            message="Amount must be positive",
            context={"amount": data.amount}
        )
    
    # Continue with calculation...
```

The API will respond with:

```json
{
  "error": "Amount must be positive",
  "type": "ValidationError",
  "timestamp": "2023-03-14T18:30:45.123456",
  "context": {
    "amount": 0
  }
}
```

### Helper Functions

Use helper functions for safer error handling:

```python
from app.core.error_handling import safely_run, safely_run_async, extract_error_info

# Run a function safely, returning None on error
result = safely_run(complex_calculation, data)
if result is None:
    # Handle failure case

# For async functions
result = await safely_run_async(async_complex_calculation, data)

# Extract error information
try:
    result = risky_operation()
except Exception as e:
    error_info = extract_error_info(e)
    logger.error("Operation failed", extra=error_info)
```

## Best Practices

### 1. Use the Right Error Type

Choose the most specific error type for the issue:

- `CalculationError`: For financial calculation failures
- `DataError`: For data validation or parsing issues
- `ServiceError`: For external service failures (APIs, etc.)
- `CacheError`: For Redis or in-memory cache issues
- `DatabaseError`: For database operation failures
- `ValidationError`: For input validation errors
- `ConfigurationError`: For configuration/setup problems

### 2. Include Contextual Information

Always provide relevant context with errors:

```python
raise ServiceError(
    message="Failed to fetch market data",
    context={
        "market_id": market_id,
        "timestamp": request_time,
        "endpoint": endpoint_url
    }
)
```

### 3. Preserve Original Exceptions

When wrapping an exception, always include the original as the cause:

```python
try:
    response = await http_client.get(url)
    data = response.json()
except Exception as e:
    raise ServiceError(
        message="API request failed",
        context={"url": url},
        cause=e  # Original exception preserved
    )
```

### 4. Use Decorators for Consistency

Apply the `@handle_errors` decorator to service methods and API handlers for consistent error handling:

```python
@handle_errors(
    logger=logger,
    error_mapping={
        redis.RedisError: CacheError,
        json.JSONDecodeError: DataError,
        requests.RequestException: ServiceError
    }
)
async def fetch_and_cache_data(key, url):
    # Implementation...
```

### 5. Log Errors at Appropriate Levels

- Use ERROR level for exceptions that need attention
- Use WARNING for handled exceptions that don't need immediate attention
- Include enough context for debugging but avoid sensitive data

### 6. Handle Errors at Boundaries

Handle errors at service and API boundaries:

- Services should use specific ApplicationError types
- APIs should use the registered exception handlers
- Don't let raw exceptions propagate to users

## Production Considerations

### Security

- In production, internal error details are hidden from API responses
- Sensitive data is filtered from logged errors
- Stack traces are only visible in logs, not API responses

### Observability

- All ApplicationErrors are logged with context
- Error hierarchies allow filtering and aggregation in logging systems
- Integration with Sentry captures detailed error information

### Performance

- The error handling framework adds minimal overhead
- Context information is only serialized when errors occur
- Helper functions like `safely_run` have negligible performance impact

## Example Implementations

### Example 1: Error Handling in Redis Caching

```python
from app.core.error_handling import handle_errors, CacheError
import json
import redis

@handle_errors(
    error_mapping={
        redis.RedisError: CacheError,
        json.JSONDecodeError: CacheError
    }
)
async def get_cached_data(key):
    try:
        data = await redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    except redis.ConnectionError as e:
        # Special handling for connection issues
        logger.warning(f"Redis connection error: {e}")
        
        # Attempt fallback to in-memory cache
        if key in memory_cache:
            return memory_cache[key]
            
        # Re-raise with context for proper handling
        raise CacheError(
            message="Redis connection failed, fallback cache miss",
            context={"key": key},
            cause=e
        )
```

### Example 2: Error Handling in Financial Calculations

```python
from app.core.error_handling import handle_errors, CalculationError

@handle_errors(
    error_mapping={
        ZeroDivisionError: CalculationError,
        ValueError: CalculationError
    },
    context={"calculation_type": "npv"}
)
def calculate_npv(cashflows, rate):
    if not cashflows:
        raise CalculationError(
            message="No cashflows provided for NPV calculation",
            context={"rate": rate}
        )
        
    if rate <= -1:
        raise CalculationError(
            message="Invalid discount rate for NPV calculation",
            context={"rate": rate}
        )
        
    # Calculate NPV...
    # Any exceptions will be caught and converted to CalculationError
```

### Example 3: API Endpoint with Error Handling

```python
from fastapi import APIRouter, Depends
from app.core.error_handling import handle_errors, ValidationError, ServiceError
from app.models.request import SimulationRequest
from app.services.simulation import SimulationService

router = APIRouter()

@router.post("/simulations")
@handle_errors(
    error_mapping={
        ValueError: ValidationError,
        # Other mappings...
    }
)
async def create_simulation(
    request: SimulationRequest,
    simulation_service: SimulationService = Depends()
):
    # Validate request
    if request.num_scenarios < 1:
        raise ValidationError(
            message="Number of scenarios must be at least 1",
            context={"num_scenarios": request.num_scenarios}
        )
        
    # Run simulation
    try:
        result = await simulation_service.run_simulation(request)
        return {"simulation_id": result.id, "status": "completed"}
    except ServiceError as e:
        # The decorator will handle this ApplicationError properly
        raise
```

## Troubleshooting

### Common Issues and Solutions

1. **Error Not Showing in API Response**
   - Check that the error is an `ApplicationError` subclass
   - Verify exception handlers are registered in `main.py`

2. **Missing Context in Logs**
   - Ensure you're providing context in the `raise` statement
   - Check if the logger is configured correctly

3. **Decorator Not Working**
   - Verify the function signature matches the expected parameters
   - Check for syntax errors in the decorator application
   - Make sure async functions have `await` applied correctly

4. **Performance Issues**
   - Reduce context data size for errors in hot paths
   - Use more specific error mappings to avoid expensive isinstance checks

## Migration Guide

When migrating existing code to use the new error handling framework:

1. Replace raw exceptions with appropriate `ApplicationError` subclasses
2. Apply the `@handle_errors` decorator to functions
3. Update error handling in API endpoints to use the new classes
4. Refactor try/except blocks to include proper context and cause information

## Additional Resources

- [Python Error Handling Best Practices](https://docs.python.org/3/tutorial/errors.html)
- [FastAPI Exception Handling](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [Sentry Python Integration](https://docs.sentry.io/platforms/python/)

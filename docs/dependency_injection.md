# Dependency Injection Pattern in Credit Cashflow Engine

## Overview

This document explains the dependency injection (DI) pattern used in the Credit Cashflow Engine application. The DI pattern is a software design pattern that promotes loose coupling between components by allowing dependencies to be injected rather than created internally. This approach significantly improves testability, maintainability, and flexibility.

## Core Benefits

By adopting dependency injection throughout our application, we gain:

1. **Standardized service initialization** - All services are created and configured in a consistent manner
2. **Improved testability** - Services can be easily mocked for unit testing
3. **Reduced coupling** - Services don't create their own dependencies
4. **Centralized configuration** - Service dependencies are managed in one place
5. **Runtime flexibility** - Services can be swapped without code changes
6. **Clear dependency hierarchy** - The dependency graph is explicit and visible

## Implementation in Credit Cashflow Engine

We've implemented dependency injection using a centralized `ServiceContainer` that manages the lifecycle of services. The container supports:

1. Singleton services
2. Factory-based service creation
3. Automatic dependency resolution
4. Scoped service instances

### Key Components

- **`dependency_injection.py`**: The core DI container implementation
- **`service_registry.py`**: The composition root where services are registered
- **`container_deps.py`**: FastAPI integration for dependency resolution

### Usage Patterns

#### 1. Service Definition

Services should be defined with explicit dependencies in their constructors:

```python
class AnalyticsService:
    def __init__(
        self, 
        absbox_service: AbsBoxServiceEnhanced,
        cache_service: CacheService,
        cache_ttl: int = 3600
    ):
        self.absbox = absbox_service
        self.cache_service = cache_service
        self.cache_ttl = cache_ttl
```

#### 2. Service Registration

Services are registered in the `service_registry.py` module:

```python
# Register a singleton instance
def register_database_client():
    database_client = SupabaseClient()
    register_service(SupabaseClient, database_client)

# Register a factory function for on-demand creation
def register_absbox_service():
    def create_absbox_service():
        cache_service = container.resolve(CacheService)
        return AbsBoxServiceEnhanced(
            cache_service=cache_service,
            redis_url=settings.REDIS_URL,
        )
    
    register_factory(AbsBoxServiceEnhanced, create_absbox_service)
```

#### 3. Service Resolution in FastAPI

Services are injected into API endpoints using FastAPI's dependency system:

```python
from app.api.container_deps import get_absbox_service, get_cache_service

@router.post("/analytics/")
async def calculate_analytics(
    request: AnalyticsRequest,
    absbox_service: AbsBoxServiceEnhanced = Depends(get_absbox_service),
    cache_service: CacheService = Depends(get_cache_service)
):
    # Use the injected services
    ...
```

#### 4. Direct Resolution

Services can also be resolved directly from the container:

```python
from app.core.dependency_injection import container

# Resolve a service
cache_service = container.resolve(CacheService)
```

## Best Practices

### 1. Constructor Injection

Always use constructor injection instead of setter injection or property injection:

```python
# GOOD - Constructor injection
def __init__(self, cache_service: CacheService):
    self.cache_service = cache_service

# BAD - Property injection
def __init__(self):
    self.cache_service = None  # Set later

def set_cache_service(self, cache_service: CacheService):
    self.cache_service = cache_service
```

### 2. Interface-Based Dependencies

Define services in terms of interfaces (abstract base classes) rather than concrete implementations:

```python
# GOOD - Depends on interface
def __init__(self, cache_service: CacheServiceInterface):
    self.cache_service = cache_service

# BAD - Depends on concrete implementation
def __init__(self, cache_service: RedisCacheService):
    self.cache_service = cache_service
```

### 3. Single Responsibility

Each service should have a single, well-defined responsibility:

```python
# GOOD - Single responsibility
class CacheService:
    def get(self, key: str): ...
    def set(self, key: str, value: Any): ...

# BAD - Multiple responsibilities
class DataService:
    def get_from_cache(self, key: str): ...
    def set_to_cache(self, key: str, value: Any): ...
    def fetch_from_database(self, query: str): ...
    def update_database(self, data: Dict): ...
```

### 4. Explicit Dependencies

Make all dependencies explicit in the constructor:

```python
# GOOD - Explicit dependencies
def __init__(self, logger: Logger, config: Config):
    self.logger = logger
    self.config = config

# BAD - Hidden dependencies
def __init__(self):
    self.logger = logging.getLogger(__name__)
    self.config = load_config_from_file()
```

### 5. Avoid Service Locator Pattern

Don't use the service container as a service locator within services:

```python
# GOOD - Dependencies injected
def __init__(self, cache_service: CacheService):
    self.cache_service = cache_service

# BAD - Service locator pattern
def process_data(self):
    cache_service = container.resolve(CacheService)
    cache_service.get("data")
```

### 6. Test with Mocks

In tests, use mock implementations of dependencies:

```python
def test_analytics_service():
    # Create mocks
    mock_absbox_service = MockAbsBoxService()
    mock_cache_service = MockCacheService()
    
    # Inject mocks
    service = AnalyticsService(
        absbox_service=mock_absbox_service,
        cache_service=mock_cache_service
    )
    
    # Test with controlled mocks
    result = service.calculate_metrics(test_cashflows, 0.05)
    assert result.npv == expected_npv
```

## Production-Ready Features

Our dependency injection system includes several production-ready features to ensure reliability, performance, and robustness:

### 1. Graceful Error Handling and Fallbacks

All service registrations include proper error handling with graceful fallbacks:

```python
def register_cache_service():
    """Register cache service with proper error handling and fallback mechanisms"""
    try:
        # Get Redis config from container
        redis_config = container.resolve(RedisConfig)
        
        # Create cache service with proper config
        cache_service = CacheService(redis_config=redis_config)
        register_service(CacheService, cache_service)
        
    except Exception as e:
        logger.error(f"Failed to register cache service: {str(e)}")
        # Create fallback in-memory-only cache
        fallback_cache = CacheService(redis_config=None)
        register_service(CacheService, fallback_cache)
        logger.warning("Using fallback in-memory-only cache service")
```

This ensures that even if Redis is unavailable, the application can continue functioning with reduced capabilities rather than failing completely.

### 2. Service Initialization Performance Monitoring

Service initialization is monitored for performance to identify potential bottlenecks:

```python
@contextmanager
def timed_init(service_name: str):
    """Context manager to time service initialization"""
    start_time = time.time()
    try:
        yield
    finally:
        if METRICS_ENABLED:
            initialization_time = time.time() - start_time
            DEPENDENCY_INIT_TIME.labels(service=service_name).observe(initialization_time)
            logger.debug(f"Initialized {service_name} in {initialization_time:.3f}s")

# Usage
with timed_init("cache_service"):
    register_cache_service()
```

### 3. Comprehensive Error Classification

All service errors are properly classified and contextualized for easier troubleshooting:

```python
class RegistryError(ApplicationError):
    """Error during service registration"""
    pass

# Usage
except Exception as e:
    raise RegistryError(
        message="Failed to initialize database client",
        context={"supabase_url": settings.SUPABASE_URL},
        cause=e
    )
```

### 4. Environment-Specific Configuration

Services are configured differently based on the environment:

```python
# Production settings
if settings.ENVIRONMENT == "production":
    max_memory_items = 10000
    socket_timeout = 5.0
    max_retries = 3
# Development settings
else:
    max_memory_items = 1000
    socket_timeout = 10.0
    max_retries = 1
```

### 5. Redis Integration with Circuit Breakers

The Redis cache implementation includes circuit-breaker patterns to prevent cascading failures:

```python
async def _verify_redis_connection(self) -> bool:
    """Verify Redis connection is working"""
    if not self.redis_client:
        return False
    
    try:
        await self.redis_client.ping()
        if not self.redis_available:
            self.logger.info("Redis connection restored")
            self.redis_available = True
        return True
    except Exception as e:
        if self.redis_available:
            self.logger.warning(f"Redis connection lost: {str(e)}")
            self.redis_available = False
        return False
```

When Redis becomes unavailable, the service automatically falls back to in-memory caching without impacting application performance.

### 6. Centralized Service Configuration

All service configuration is centralized in the `service_registry.py` module, making it easy to adjust settings in one place:

```python
def _create_redis_config() -> RedisConfig:
    """Create Redis configuration with appropriate settings"""
    redis_url = _get_redis_url()
    
    return RedisConfig(
        url=redis_url,
        socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
        socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
        retry_on_timeout=settings.REDIS_RETRY_ON_TIMEOUT,
        max_connections=settings.REDIS_MAX_CONNECTIONS,
        health_check_interval=settings.REDIS_HEALTH_CHECK_INTERVAL,
        max_retries=settings.REDIS_RETRY_MAX_ATTEMPTS,
    )
```

## Troubleshooting

### Common Issues

#### 1. Circular Dependencies

**Problem**: Service A depends on Service B, which depends on Service A.

**Solution**: 
- Refactor to eliminate circular dependency by introducing a mediator service
- Use a callback or event system to decouple the services
- Delay the resolution of one dependency until needed

#### 2. Missing Dependency Registration

**Problem**: Trying to resolve a service that hasn't been registered.

**Solution**:
- Check `service_registry.py` to ensure all services are registered
- Use the `container.has(ServiceType)` method to check for registration
- Add proper error handling when resolving services

#### 3. Service Configuration Errors

**Problem**: Service fails to initialize due to incorrect configuration.

**Diagnostics**:
- Check the logs for specific error messages
- Look for `ConfigurationError` or `ServiceError` exceptions
- Verify environment variables are set correctly

**Solution**:
- Correct the configuration values
- Add fallback configurations for non-critical services

#### 4. Redis Connection Issues

**Problem**: Redis cache service fails to connect.

**Diagnostics**:
- Look for "Failed to initialize Redis client" or "Redis connection lost" log messages
- Check if `REDIS_URL` or other Redis settings are configured correctly
- Verify Redis server is running and accessible

**Solution**:
- Verify Redis connection settings
- Check network connectivity
- The system will automatically fallback to in-memory cache

## Performance Considerations

1. **Service Resolution Caching**: Services are cached after first resolution to avoid reconstruction costs.

2. **Lazy Initialization**: Factory-based services are only initialized when first needed.

3. **Resource Management**: Connection pools and shared resources are properly configured for optimal performance.

4. **Memory Footprint**: In-memory caches are configured with appropriate max items to prevent memory leaks.

5. **Timeout Handling**: All external service connections have appropriate timeout settings to prevent hanging threads.

## Testing the Dependency Injection System

Comprehensive tests are included in `tests/core/test_dependency_injection.py` to verify:

1. Service container initialization
2. Service registration and resolution
3. Proper dependency injection
4. Fallback mechanisms when services are unavailable 
5. Error handling in service initialization
6. Thread safety for concurrent access

Run the tests with:

```bash
pytest tests/core/test_dependency_injection.py -v
```

To run Redis integration tests (requires a Redis server):

```bash
REDIS_INTEGRATION_TEST=true pytest tests/core/test_dependency_injection.py -vk redis
```

## Adding a New Service

To add a new service to the application:

1. Define the service class with explicit constructor dependencies
2. Register the service in `service_registry.py`
3. Add a FastAPI dependency function in `container_deps.py`
4. Use the dependency in your API endpoints

```python
# 1. Define service
class ReportingService:
    def __init__(self, absbox_service: AbsBoxServiceEnhanced, db: SupabaseClient):
        self.absbox_service = absbox_service
        self.db = db

# 2. Register in service_registry.py
def register_reporting_service():
    def create_reporting_service():
        absbox_service = container.resolve(AbsBoxServiceEnhanced)
        db = container.resolve(SupabaseClient)
        return ReportingService(absbox_service=absbox_service, db=db)
    
    register_factory(ReportingService, create_reporting_service)

# 3. Add to container_deps.py
get_reporting_service = get_service(ReportingService)

# 4. Use in API endpoints
@router.get("/reports/")
async def get_reports(
    reporting_service: ReportingService = Depends(get_reporting_service)
):
    return reporting_service.generate_reports()
```

## Conclusion

By following these patterns and best practices, we'll maintain a clean, testable, and maintainable codebase. Dependency injection helps us create loosely coupled components that are easier to develop, test, and maintain over time.

For questions or improvements to this pattern, please contact the core development team.

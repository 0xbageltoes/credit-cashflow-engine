# Credit Cashflow Engine Service Consolidation Plan

## 1. Unified Redis Service

The current codebase has multiple Redis implementations spread across different files:
- `app/core/redis_cache.py` - Basic Redis cache wrapper with singleton pattern
- `app/services/redis_client.py` - More robust Redis client with error handling
- `app/services/redis_service.py` - Comprehensive Redis service with sync/async methods

### Consolidation Strategy

1. Create a single unified `RedisService` class that incorporates:
   - Comprehensive configuration via a `RedisConfig` class
   - Both synchronous and asynchronous client support
   - Robust error handling and connection recovery
   - Consistent serialization/deserialization methods
   - Performance monitoring via Prometheus metrics
   - Proper connection pooling

2. Implement connection resilience features:
   - Graceful degradation when Redis is unavailable
   - Automatic reconnection logic
   - Circuit breaker pattern to prevent cascading failures
   - Detailed logging for troubleshooting

3. Standardize caching strategies:
   - Deterministic cache key generation
   - Consistent TTL handling
   - Hierarchical caching (memory → Redis)
   - Proper cache invalidation

## 2. Consolidated AbsBox Service

Current implementation has two versions:
- `app/services/absbox_service.py` - Basic service with core functionality
- `app/services/absbox_service_enhanced.py` - Enhanced version with more features

### Consolidation Strategy

1. Create a unified `AbsBoxService` class that:
   - Combines all functionality from both existing services
   - Uses consistent dependency injection
   - Implements proper async/await patterns
   - Utilizes the unified Redis service
   - Provides comprehensive error handling

2. Architectural improvements:
   - Separate interface and implementation for better testability
   - Create specialized sub-services for specific calculation domains
   - Implement proper factory patterns for object creation

3. Performance optimizations:
   - Vectorize calculations using NumPy
   - Implement efficient batch processing
   - Optimize the Monte Carlo simulation engine

## 3. Implementation Plan

### Phase 1: Unified Redis Service (1-2 weeks)
1. Create the consolidated `RedisService` class
2. Implement comprehensive test suite
3. Update existing services to use the new Redis service

### Phase 2: AbsBox Service Consolidation (2-3 weeks)
1. Create the consolidated `AbsBoxService` class
2. Implement domain-specific calculation services
3. Develop comprehensive test suite

### Phase 3: Performance Optimization (1-2 weeks)
1. Vectorize numerical calculations
2. Implement batch processing
3. Performance testing and optimization

### Phase 4: Documentation and Examples (1 week)
1. Document all public interfaces
2. Create usage examples
3. Update API documentation

## 4. Migration Strategy

1. Implement new services alongside existing ones
2. Add feature flags to control service routing
3. Gradually migrate API endpoints to new implementations
4. Run extensive tests to ensure functional equivalence
5. Remove deprecated services once migration is complete

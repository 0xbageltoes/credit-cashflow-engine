"""
Tests for the dependency injection system in the Credit Cashflow Engine.

This module verifies:
1. Service container initialization
2. Service registration and resolution
3. Proper dependency injection with Redis integration
4. Fallback mechanisms when services are unavailable
5. Error handling in service initialization
"""
import pytest
import time
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from app.core.dependency_injection import (
    ServiceContainer, 
    container,
    register_service,
    register_factory,
    resolve_service,
    create_container_scope
)
from app.core.error_handling import ServiceError, CacheError, ApplicationError
from app.core.cache_service import CacheService, RedisConfig
from app.core.service_registry import register_all_services, _create_redis_config, register_cache_service
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced
from app.services.asset_handlers.consumer_credit import ConsumerCreditHandler


class TestServiceContainer:
    """Tests for the ServiceContainer class functionality"""
    
    def test_container_initialization(self):
        """Test that a container can be properly initialized"""
        container = ServiceContainer()
        assert container._services == {}
        assert container._factories == {}
    
    def test_service_registration(self):
        """Test registering a service instance"""
        container = ServiceContainer()
        
        # Create a simple test service
        class TestService:
            def __init__(self, value: str = "test"):
                self.value = value
        
        # Register the service
        test_service = TestService("test_value")
        container.register(TestService, test_service)
        
        # Verify it was registered correctly
        assert TestService in container._services
        assert container._services[TestService] is test_service
    
    def test_factory_registration(self):
        """Test registering a factory function"""
        container = ServiceContainer()
        
        # Create a simple test service
        class TestService:
            def __init__(self, value: str = "test"):
                self.value = value
        
        # Create a factory function
        def factory():
            return TestService("factory_value")
        
        # Register the factory
        container.register_factory(TestService, factory)
        
        # Verify it was registered correctly
        assert TestService in container._factories
        assert container._factories[TestService] is factory
    
    def test_service_resolution(self):
        """Test resolving a registered service"""
        container = ServiceContainer()
        
        # Create a simple test service
        class TestService:
            def __init__(self, value: str = "test"):
                self.value = value
        
        # Register the service
        test_service = TestService("test_value")
        container.register(TestService, test_service)
        
        # Resolve the service
        resolved = container.resolve(TestService)
        
        # Verify it was resolved correctly
        assert resolved is test_service
        assert resolved.value == "test_value"
    
    def test_factory_resolution(self):
        """Test resolving a service via factory"""
        container = ServiceContainer()
        
        # Create a simple test service
        class TestService:
            def __init__(self, value: str = "test"):
                self.value = value
        
        # Create a factory function
        def factory():
            return TestService("factory_value")
        
        # Register the factory
        container.register_factory(TestService, factory)
        
        # Resolve the service
        resolved = container.resolve(TestService)
        
        # Verify it was resolved correctly
        assert isinstance(resolved, TestService)
        assert resolved.value == "factory_value"
        
        # Verify the instance is now cached
        assert TestService in container._services
        assert container._services[TestService] is resolved
    
    def test_auto_resolve_with_dependencies(self):
        """Test auto-resolving a service with dependencies"""
        container = ServiceContainer()
        
        # Create dependent services
        class DependencyA:
            def __init__(self):
                self.name = "DependencyA"
        
        class DependencyB:
            def __init__(self, dep_a: DependencyA):
                self.dep_a = dep_a
                self.name = "DependencyB"
        
        class TestService:
            def __init__(self, dep_a: DependencyA, dep_b: DependencyB):
                self.dep_a = dep_a
                self.dep_b = dep_b
                self.name = "TestService"
        
        # Register only DependencyA
        dep_a = DependencyA()
        container.register(DependencyA, dep_a)
        
        # Try to resolve TestService, which should auto-resolve DependencyB
        resolved = container.resolve(TestService)
        
        # Verify the full dependency chain was resolved
        assert isinstance(resolved, TestService)
        assert resolved.name == "TestService"
        assert resolved.dep_a is dep_a
        assert isinstance(resolved.dep_b, DependencyB)
        assert resolved.dep_b.dep_a is dep_a
    
    def test_scoped_container(self):
        """Test creating a scoped container"""
        parent = ServiceContainer()
        
        # Register a service in the parent
        class ParentService:
            def __init__(self):
                self.name = "ParentService"
        
        parent_service = ParentService()
        parent.register(ParentService, parent_service)
        
        # Create a scoped container
        scoped = parent.create_scope()
        
        # Verify the service is available in the scoped container
        resolved = scoped.resolve(ParentService)
        assert resolved is parent_service
        
        # Register a service in the scoped container
        class ScopedService:
            def __init__(self):
                self.name = "ScopedService"
        
        scoped_service = ScopedService()
        scoped.register(ScopedService, scoped_service)
        
        # Verify the scoped service is available in the scoped container
        resolved = scoped.resolve(ScopedService)
        assert resolved is scoped_service
        
        # Verify the scoped service is NOT available in the parent
        with pytest.raises(ValueError):
            parent.resolve(ScopedService)


class TestServiceRegistry:
    """Tests for the service registry integration"""
    
    @patch('app.core.service_registry.container')
    def test_redis_config_creation(self, mock_container):
        """Test Redis config creation with correct parameters"""
        # Create a Redis config
        redis_config = _create_redis_config()
        
        # Verify the config has the correct properties
        assert isinstance(redis_config, RedisConfig)
        assert hasattr(redis_config, 'url')
        assert hasattr(redis_config, 'socket_timeout')
        assert hasattr(redis_config, 'socket_connect_timeout')
        assert hasattr(redis_config, 'retry_on_timeout')
        assert hasattr(redis_config, 'max_connections')
        assert hasattr(redis_config, 'health_check_interval')
        assert hasattr(redis_config, 'max_retries')
    
    @patch('app.core.service_registry.container')
    def test_cache_service_registration_with_redis(self, mock_container):
        """Test cache service registration with Redis"""
        # Mock the Redis config
        redis_config = RedisConfig(url="redis://test:6379")
        mock_container.resolve.return_value = redis_config
        
        # Register the cache service
        register_cache_service()
        
        # Verify the cache service was registered correctly
        assert mock_container.register.called
        register_call = mock_container.register.call_args[0]
        assert register_call[0] == CacheService
        assert isinstance(register_call[1], CacheService)
    
    @patch('app.core.service_registry.container')
    def test_cache_service_fallback_without_redis(self, mock_container):
        """Test cache service registration with fallback when Redis is unavailable"""
        # Mock the container to raise an exception when resolving Redis config
        mock_container.resolve.side_effect = ValueError("Redis config not found")
        
        # Register the cache service
        register_cache_service()
        
        # Verify the cache service was registered with a fallback
        assert mock_container.register.called
        register_call = mock_container.register.call_args[0]
        assert register_call[0] == CacheService
        assert isinstance(register_call[1], CacheService)
        # The fallback service should not have Redis config
        assert register_call[1].redis_client is None


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the dependency injection system"""
    
    def setup_method(self):
        """Set up a fresh container for each test"""
        # Reset the global container
        global container
        container = ServiceContainer()
    
    def test_full_registration(self):
        """Test a full registration of all services"""
        # This is the main integration test that verifies all services
        # can be registered without errors
        try:
            register_all_services()
        except Exception as e:
            pytest.fail(f"Service registration failed: {str(e)}")
    
    def test_absbox_service_resolution(self):
        """Test resolving the AbsBox service with all dependencies"""
        # Register all services
        register_all_services()
        
        # Resolve the AbsBox service
        absbox_service = resolve_service(AbsBoxServiceEnhanced)
        
        # Verify the service was resolved correctly
        assert isinstance(absbox_service, AbsBoxServiceEnhanced)
        assert hasattr(absbox_service, 'cache_service')
        assert isinstance(absbox_service.cache_service, CacheService)
    
    def test_consumer_credit_handler_resolution(self):
        """Test resolving a higher-level service with nested dependencies"""
        # Register all services
        register_all_services()
        
        # Resolve the handler
        handler = resolve_service(ConsumerCreditHandler)
        
        # Verify the handler and its dependencies
        assert isinstance(handler, ConsumerCreditHandler)
        assert hasattr(handler, 'absbox_service')
        assert isinstance(handler.absbox_service, AbsBoxServiceEnhanced)
    
    def test_concurrent_resolution(self):
        """Test resolving services concurrently to verify thread safety"""
        import threading
        
        # Register all services
        register_all_services()
        
        # Define the worker function
        results = []
        exceptions = []
        
        def worker():
            try:
                # Resolve a service in each thread
                service = resolve_service(AbsBoxServiceEnhanced)
                results.append(service)
            except Exception as e:
                exceptions.append(e)
        
        # Create and start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no exceptions occurred
        assert not exceptions, f"Exceptions occurred during concurrent resolution: {exceptions}"
        
        # Verify all threads got the same instance (singleton behavior)
        assert len(results) == 10
        assert all(result is results[0] for result in results)


@pytest.mark.redis
class TestRedisIntegration:
    """Tests for Redis integration with dependency injection
    
    These tests require a Redis server to be running and configured.
    They will be skipped if the REDIS_INTEGRATION_TEST environment variable is not set.
    """
    
    @pytest.fixture(autouse=True)
    def check_redis_integration(self):
        """Skip tests if Redis integration is not enabled"""
        if os.environ.get('REDIS_INTEGRATION_TEST') != 'true':
            pytest.skip("Redis integration tests disabled. Set REDIS_INTEGRATION_TEST=true to enable.")
    
    def setup_method(self):
        """Set up a fresh container for each test"""
        # Reset the global container
        global container
        container = ServiceContainer()
        register_all_services()
    
    def test_cache_service_with_real_redis(self):
        """Test cache service with a real Redis connection"""
        # Resolve the cache service
        cache_service = resolve_service(CacheService)
        
        # Verify Redis client was created
        assert cache_service.redis_client is not None
        
        # Try to use the cache
        test_key = f"test:di:{time.time()}"
        test_value = {"test": "value", "timestamp": time.time()}
        
        # Set a value
        cache_service.set(test_key, test_value)
        
        # Get the value back
        retrieved = cache_service.get(test_key)
        
        # Verify the value was stored and retrieved correctly
        assert retrieved is not None
        assert retrieved.get("test") == "value"
        
        # Clean up
        cache_service.delete(test_key)
    
    def test_cache_fallback_with_redis_error(self):
        """Test cache fallback when Redis operations fail"""
        # Resolve the cache service
        cache_service = resolve_service(CacheService)
        
        # Force Redis client to raise an exception
        original_get = cache_service.redis_client.get
        
        async def mock_get(*args, **kwargs):
            raise Exception("Redis error")
        
        cache_service.redis_client.get = mock_get
        
        # Try to use the cache - should fall back to memory cache
        test_key = f"test:di:fallback:{time.time()}"
        test_value = {"test": "fallback", "timestamp": time.time()}
        
        # Add to memory cache directly
        cache_service.memory_cache[test_key] = test_value
        cache_service.expiry_times[test_key] = datetime.now() + timedelta(minutes=5)
        
        # Get the value - should come from memory cache despite Redis error
        retrieved = cache_service.get(test_key)
        
        # Verify the fallback worked
        assert retrieved is not None
        assert retrieved.get("test") == "fallback"
        
        # Restore original method
        cache_service.redis_client.get = original_get


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

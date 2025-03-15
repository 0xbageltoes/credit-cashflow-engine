"""
Service Registry

This module registers all application services in the dependency injection container.
It acts as a composition root for the application, centralizing service initialization and configuration.
"""

import logging
from typing import Optional, Dict, Any, Type, TypeVar, cast
import os
import time
import json
from functools import wraps
from contextlib import contextmanager

from app.core.dependency_injection import container, register_service, register_factory
from app.core.config import settings, RedisConfig as ConfigRedisConfig
from app.core.error_handling import CacheError, ConfigurationError, ServiceError, handle_errors, ApplicationError
from app.core.metrics import CACHE_ERROR_COUNTER, DEPENDENCY_INIT_TIME, METRICS_ENABLED

# Import services
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced
from app.services.asset_handlers.consumer_credit import ConsumerCreditHandler
from app.services.asset_handlers.commercial_loan import CommercialLoanHandler
from app.services.asset_handlers.clo_cdo import CLOCDOHandler
from app.database.supabase import SupabaseClient
from app.core.cache_service import CacheService, RedisConfig

# Setup logger
logger = logging.getLogger(__name__)

# Type variable for generic type hints
T = TypeVar('T')

class RegistryError(ApplicationError):
    """Error during service registration"""
    pass

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


@handle_errors(logger=logger, default_error=ConfigurationError)
def register_all_services() -> None:
    """Register all application services in the dependency injection container.
    
    This is the application's composition root where all services are created and configured.
    It should be called once at application startup.
    """
    logger.info("Registering services in dependency injection container")
    
    # Register configuration services
    with timed_init("configuration"):
        register_core_configurations()
    
    # Register core services
    with timed_init("core_services"):
        register_cache_service()
        register_database_client()
    
    # Register business services
    with timed_init("business_services"):
        register_absbox_service()
        register_asset_handlers()
    
    logger.info("Service registration complete")


def register_core_configurations() -> None:
    """Register core configuration objects in the container"""
    # Configure Redis settings with proper environment-specific fallbacks
    try:
        redis_config = _create_redis_config()
        register_service(RedisConfig, redis_config)
        logger.info(f"Registered Redis configuration for environment: {settings.ENVIRONMENT}")
    except Exception as e:
        # Log the error but don't fail - we should be able to continue without Redis in degraded mode
        logger.error(f"Failed to register Redis configuration: {str(e)}")
        logger.warning("Will fallback to in-memory cache only")
        # Register a minimal config for services that require it, but it won't be used for actual Redis connections
        register_service(RedisConfig, RedisConfig(redis_url="redis://localhost:6379"))


def _create_redis_config() -> RedisConfig:
    """Create Redis configuration with appropriate settings for the current environment"""
    # Get Redis URL with appropriate fallbacks
    redis_url = _get_redis_url()
    
    # Create config with all the appropriate settings
    return RedisConfig(
        url=redis_url,
        socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
        socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
        retry_on_timeout=settings.REDIS_RETRY_ON_TIMEOUT,
        max_connections=settings.REDIS_MAX_CONNECTIONS,
        health_check_interval=settings.REDIS_HEALTH_CHECK_INTERVAL,
        max_retries=settings.REDIS_RETRY_MAX_ATTEMPTS,
    )


def _get_redis_url() -> str:
    """Get Redis URL with appropriate fallbacks based on environment"""
    # Use explicit REDIS_URL if provided
    if settings.REDIS_URL:
        return settings.REDIS_URL
    
    # Otherwise build from components
    protocol = "rediss" if settings.REDIS_SSL else "redis"
    auth = f":{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else ""
    return f"{protocol}://{auth}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"


def register_cache_service() -> None:
    """Register cache service with proper error handling and fallback mechanisms"""
    try:
        # Get Redis config from container if available
        redis_config = None
        try:
            redis_config = container.resolve(RedisConfig)
        except Exception as e:
            logger.warning(f"Failed to resolve Redis configuration: {str(e)}")
            logger.warning("Cache service will operate with in-memory cache only")
        
        # Configure cache service with environment-specific settings
        memory_ttl = 300  # 5 minutes default
        default_ttl = settings.CACHE_TTL
        max_memory_items = 10000 if settings.ENVIRONMENT == "production" else 1000
        
        # Create cache service with proper config and fallbacks
        cache_service = CacheService(
            redis_config=redis_config,
            memory_ttl=memory_ttl,
            default_ttl=default_ttl,
            max_memory_items=max_memory_items,
            logger=logger
        )
        
        register_service(CacheService, cache_service)
        logger.info(f"Registered CacheService with Redis: {redis_config is not None}")
    except Exception as e:
        # This is more serious - log error but create a minimal in-memory-only cache
        # so the application can continue in degraded mode
        logger.error(f"Failed to register cache service: {str(e)}")
        fallback_cache = CacheService(redis_config=None, logger=logger)
        register_service(CacheService, fallback_cache)
        logger.warning("Using fallback in-memory-only cache service")


def register_database_client() -> None:
    """Register database client with proper error handling"""
    try:
        # Create database client with proper configuration
        database_client = SupabaseClient(
            supabase_url=settings.SUPABASE_URL,
            supabase_key=settings.SUPABASE_ANON_KEY or settings.SUPABASE_KEY,
            supabase_service_key=settings.SUPABASE_SERVICE_KEY,
        )
        register_service(SupabaseClient, database_client)
        logger.info("Registered SupabaseClient")
    except Exception as e:
        logger.error(f"Failed to register database client: {str(e)}")
        raise RegistryError(
            message="Failed to initialize database client",
            context={"supabase_url": settings.SUPABASE_URL},
            cause=e
        )


def register_absbox_service() -> None:
    """Register AbsBox service with proper dependencies and error handling"""
    # Create service factory to support both async and sync initialization
    def create_absbox_service():
        try:
            # Resolve required dependencies from container
            cache_service = container.resolve(CacheService)
            
            # Get additional configuration from settings
            absbox_config = {
                "cache_service": cache_service,
                "redis_url": settings.REDIS_URL,
                "absbox_url": settings.ABSBOX_SERVICE_URL,
                "absbox_api_key": settings.ABSBOX_API_KEY,
                "use_cache": settings.USE_REDIS_CACHE,
                "cache_ttl": settings.CACHE_TTL,
                "max_retries": settings.ABSBOX_MAX_RETRIES,
                "retry_delay": settings.ABSBOX_RETRY_DELAY,
                "timeout": settings.ABSBOX_TIMEOUT,
            }
            
            # Create service with proper configuration
            return AbsBoxServiceEnhanced(**absbox_config)
        except Exception as e:
            logger.error(f"Failed to create AbsBoxServiceEnhanced: {str(e)}")
            raise ServiceError(
                message="Failed to initialize AbsBox service",
                context={
                    "absbox_url": settings.ABSBOX_SERVICE_URL,
                    "cache_enabled": settings.USE_REDIS_CACHE,
                },
                cause=e
            )
    
    register_factory(AbsBoxServiceEnhanced, create_absbox_service)
    logger.info("Registered AbsBoxServiceEnhanced factory")


def register_asset_handlers() -> None:
    """Register all asset handlers with proper dependencies and error handling"""
    # Register consumer credit handler factory
    def create_consumer_credit_handler():
        try:
            absbox_service = container.resolve(AbsBoxServiceEnhanced)
            return ConsumerCreditHandler(absbox_service=absbox_service)
        except Exception as e:
            logger.error(f"Failed to create ConsumerCreditHandler: {str(e)}")
            raise ServiceError(
                message="Failed to initialize Consumer Credit handler",
                cause=e
            )
    
    register_factory(ConsumerCreditHandler, create_consumer_credit_handler)
    
    # Register commercial loan handler factory
    def create_commercial_loan_handler():
        try:
            absbox_service = container.resolve(AbsBoxServiceEnhanced)
            return CommercialLoanHandler(absbox_service=absbox_service)
        except Exception as e:
            logger.error(f"Failed to create CommercialLoanHandler: {str(e)}")
            raise ServiceError(
                message="Failed to initialize Commercial Loan handler",
                cause=e
            )
    
    register_factory(CommercialLoanHandler, create_commercial_loan_handler)
    
    # Register CLO/CDO handler factory
    def create_clo_cdo_handler():
        try:
            absbox_service = container.resolve(AbsBoxServiceEnhanced)
            return CLOCDOHandler(absbox_service=absbox_service)
        except Exception as e:
            logger.error(f"Failed to create CLOCDOHandler: {str(e)}")
            raise ServiceError(
                message="Failed to initialize CLO/CDO handler",
                cause=e
            )
    
    register_factory(CLOCDOHandler, create_clo_cdo_handler)
    
    logger.info("Registered asset handler factories")

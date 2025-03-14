"""
Dependency Injection Container

This module provides a unified approach to dependency injection throughout the application.
It helps standardize service creation, improve testability, and reduce tight coupling.
"""
from typing import Dict, Type, Any, Optional, TypeVar, cast, Callable
import inspect
import logging

T = TypeVar('T')
ServiceFactory = Callable[[], T]

class ServiceContainer:
    """Service container for dependency management
    
    This container manages the lifecycle of services, allowing for:
    - Service registration (singleton instances)
    - Factory registration (for on-demand creation)
    - Automatic dependency resolution
    - Scoped service resolution
    
    The container is designed to be used as a singleton throughout the application.
    """
    
    def __init__(self):
        """Initialize a new service container"""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, ServiceFactory] = {}
        self._logger = logging.getLogger(__name__)
    
    def register(self, service_type: Type[T], instance: T) -> None:
        """Register a service instance
        
        Args:
            service_type: Type of the service (usually an interface/abstract class)
            instance: Instance of the service
        """
        self._services[service_type] = instance
        self._logger.debug(f"Registered service: {service_type.__name__}")
    
    def register_factory(self, service_type: Type[T], factory: ServiceFactory[T]) -> None:
        """Register a factory function for a service
        
        Args:
            service_type: Type of the service
            factory: Factory function that creates service instances
        """
        self._factories[service_type] = factory
        self._logger.debug(f"Registered factory: {service_type.__name__}")
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service by type
        
        Args:
            service_type: Type of service to resolve
            
        Returns:
            Instance of the requested service
            
        Raises:
            ValueError: If service cannot be resolved
        """
        # Return existing instance if registered
        if service_type in self._services:
            return cast(T, self._services[service_type])
        
        # Create from factory if registered
        if service_type in self._factories:
            instance = self._factories[service_type]()
            self._services[service_type] = instance
            return instance
        
        # Try to auto-resolve
        return self._auto_resolve(service_type)
    
    def _auto_resolve(self, service_type: Type[T]) -> T:
        """Auto-resolve a service by inspecting constructor
        
        Args:
            service_type: Type of service to resolve
            
        Returns:
            Instance of the requested service
            
        Raises:
            ValueError: If dependencies cannot be resolved
        """
        try:
            # Get constructor parameters
            sig = inspect.signature(service_type.__init__)
            params = {}
            
            # Resolve dependencies recursively
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                
                # Check if parameter has a type annotation
                if param.annotation != inspect.Parameter.empty:
                    # Try to resolve the dependency
                    try:
                        params[name] = self.resolve(param.annotation)
                    except ValueError as e:
                        # If parameter has a default value, use it
                        if param.default != inspect.Parameter.empty:
                            params[name] = param.default
                        else:
                            raise ValueError(
                                f"Failed to resolve dependency '{name}' of type "
                                f"'{param.annotation.__name__}' for {service_type.__name__}"
                            ) from e
                elif param.default != inspect.Parameter.empty:
                    # Use default value
                    params[name] = param.default
                else:
                    raise ValueError(
                        f"Parameter '{name}' in {service_type.__name__} has no type annotation "
                        "and no default value. Cannot auto-resolve."
                    )
            
            # Create instance
            self._logger.debug(f"Auto-resolving service: {service_type.__name__}")
            instance = service_type(**params)
            
            # Register for future use
            self._services[service_type] = instance
            return instance
            
        except Exception as e:
            raise ValueError(f"Failed to auto-resolve {service_type.__name__}: {str(e)}")
    
    def create_scope(self) -> 'ServiceContainer':
        """Create a new scoped container
        
        This creates a new container that inherits all registrations from the parent,
        but allows for scoped service registrations that don't affect the parent.
        
        Returns:
            New scoped service container
        """
        scoped_container = ServiceContainer()
        
        # Copy service registrations by reference (important for singletons)
        for service_type, instance in self._services.items():
            scoped_container._services[service_type] = instance
        
        # Copy factory registrations
        for service_type, factory in self._factories.items():
            scoped_container._factories[service_type] = factory
        
        return scoped_container


# Global service container instance
container = ServiceContainer()


def register_service(service_type: Type[T], instance: T) -> None:
    """Register a service in the global container
    
    Args:
        service_type: Type of the service
        instance: Instance of the service
    """
    container.register(service_type, instance)


def register_factory(service_type: Type[T], factory: ServiceFactory[T]) -> None:
    """Register a factory in the global container
    
    Args:
        service_type: Type of the service
        factory: Factory function for the service
    """
    container.register_factory(service_type, factory)


def resolve_service(service_type: Type[T]) -> T:
    """Resolve a service from the global container
    
    Args:
        service_type: Type of the service to resolve
        
    Returns:
        Instance of the requested service
    """
    return container.resolve(service_type)


def create_container_scope() -> ServiceContainer:
    """Create a new scoped container from the global container
    
    Returns:
        New scoped service container
    """
    return container.create_scope()

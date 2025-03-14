"""
Supabase service for interacting with Supabase APIs

This module provides methods for interacting with Supabase Auth, Database, and Storage.
It handles authentication, user management, and database operations with proper error
handling and retry mechanisms for production use.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime
import copy

from app.core.config import settings
from app.services.redis_service import RedisService

# Setup logging
logger = logging.getLogger(__name__)

# Cache for database query results
redis_service = RedisService() if settings.REDIS_ENABLED else None

class SupabaseError(Exception):
    """Base exception for Supabase API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class SupabaseService:
    """
    Service for interacting with Supabase APIs
    
    This class provides methods for authentication, database operations,
    and file storage using Supabase APIs.
    """
    
    def __init__(self):
        """Initialize the Supabase service"""
        self.base_url = settings.SUPABASE_URL
        self.anon_key = settings.SUPABASE_ANON_KEY
        self.service_key = settings.SUPABASE_SERVICE_KEY
        self._health_check_timestamp = 0
        self._health_status = False

    async def _make_request(self, 
                          method: str, 
                          endpoint: str, 
                          data: Optional[Dict[str, Any]] = None,
                          params: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, Any]] = None,
                          auth_token: Optional[str] = None,
                          use_service_key: bool = False) -> Dict[str, Any]:
        """
        Make a request to the Supabase API with proper error handling
        
        Args:
            method: The HTTP method (GET, POST, PUT, DELETE)
            endpoint: The API endpoint (relative to base URL)
            data: The request body data
            params: The query parameters
            headers: Additional headers
            auth_token: Optional user auth token
            use_service_key: Whether to use the service key instead of anon key
            
        Returns:
            The response data
            
        Raises:
            SupabaseError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        # Prepare headers
        request_headers = {
            "Content-Type": "application/json",
            "apikey": self.service_key if use_service_key else self.anon_key
        }
        
        # Add authorization header if token provided
        if auth_token:
            request_headers["Authorization"] = f"Bearer {auth_token}"
        
        # Add custom headers
        if headers:
            request_headers.update(headers)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    json=data if data is not None else None,
                    params=params,
                    headers=request_headers,
                    timeout=10.0  # 10 second timeout
                )
                
                # Check for errors
                if response.status_code >= 400:
                    error_detail = None
                    try:
                        error_detail = response.json()
                    except:
                        error_detail = response.text
                    
                    raise SupabaseError(
                        message=f"Supabase API error: {response.status_code}",
                        status_code=response.status_code,
                        details=error_detail
                    )
                
                # Return the response data
                return response.json()
        
        except httpx.RequestError as e:
            logger.error(f"Request error to Supabase API: {str(e)}")
            raise SupabaseError(
                message=f"Request error: {str(e)}",
                details={"type": type(e).__name__}
            )
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Supabase API: {str(e)}")
            raise SupabaseError(
                message="Invalid JSON response",
                details={"type": "JSONDecodeError"}
            )
        
        except SupabaseError:
            # Re-raise Supabase errors
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error in Supabase request: {str(e)}")
            raise SupabaseError(
                message=f"Unexpected error: {str(e)}",
                details={"type": type(e).__name__}
            )

    @retry(
        retry=retry_if_exception_type(SupabaseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information from Supabase Auth
        
        Args:
            user_id: The user ID
            
        Returns:
            User information if found, None otherwise
            
        Raises:
            SupabaseError: If the request fails
        """
        # First check cache
        cache_key = f"user:{user_id}"
        
        if redis_service:
            try:
                cached_user = await redis_service.get(cache_key)
                if cached_user:
                    return json.loads(cached_user)
            except Exception as e:
                logger.error(f"Error checking user cache: {str(e)}")
                # Continue with API request if cache fails
        
        try:
            # Get user from Supabase Auth
            endpoint = f"/auth/v1/admin/users/{user_id}"
            user = await self._make_request(
                method="GET",
                endpoint=endpoint,
                use_service_key=True
            )
            
            # Cache the result
            if redis_service and user:
                try:
                    await redis_service.set(
                        cache_key,
                        json.dumps(user),
                        ttl=3600  # 1 hour
                    )
                except Exception as e:
                    logger.error(f"Error caching user: {str(e)}")
            
            return user
        
        except SupabaseError as e:
            if e.status_code == 404:
                # User not found
                logger.warning(f"User not found: {user_id}")
                return None
            else:
                logger.error(f"Error getting user: {str(e)}")
                raise
    
    async def query_db(self, 
                     table: str, 
                     select: str = "*", 
                     filters: Optional[Dict[str, Any]] = None,
                     order: Optional[str] = None,
                     limit: Optional[int] = None,
                     auth_token: Optional[str] = None,
                     use_cache: bool = True,
                     cache_ttl: int = 300) -> List[Dict[str, Any]]:
        """
        Query the Supabase database
        
        Args:
            table: The table name
            select: The columns to select (default: *)
            filters: Query filters
            order: Order by clause
            limit: Result limit
            auth_token: User auth token
            use_cache: Whether to use Redis cache
            cache_ttl: Cache TTL in seconds
            
        Returns:
            List of query results
            
        Raises:
            SupabaseError: If the query fails
        """
        # Generate cache key if using cache
        cache_key = None
        if use_cache and redis_service:
            # Create a deterministic cache key based on query parameters
            cache_key = f"db:{table}:{select}:{json.dumps(filters or {})}:{order or ''}:{limit or ''}"
            
            try:
                cached_result = await redis_service.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
            except Exception as e:
                logger.error(f"Error checking query cache: {str(e)}")
                # Continue with query if cache fails
        
        try:
            # Prepare the query
            endpoint = f"/rest/v1/{table}"
            headers = {"Prefer": "return=representation"}
            params = {"select": select}
            
            # Add filters
            if filters:
                for key, value in filters.items():
                    params[key] = value
            
            # Add order
            if order:
                params["order"] = order
            
            # Add limit
            if limit:
                params["limit"] = str(limit)
            
            # Execute the query
            result = await self._make_request(
                method="GET",
                endpoint=endpoint,
                params=params,
                headers=headers,
                auth_token=auth_token,
                use_service_key=auth_token is None
            )
            
            # Cache the result
            if cache_key and redis_service:
                try:
                    await redis_service.set(
                        cache_key,
                        json.dumps(result),
                        ttl=cache_ttl
                    )
                except Exception as e:
                    logger.error(f"Error caching query result: {str(e)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            raise
    
    async def insert_db(self, 
                      table: str, 
                      data: Union[Dict[str, Any], List[Dict[str, Any]]],
                      auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Insert data into the Supabase database
        
        Args:
            table: The table name
            data: The data to insert (single record or list of records)
            auth_token: User auth token
            
        Returns:
            The inserted data
            
        Raises:
            SupabaseError: If the insertion fails
        """
        try:
            # Prepare the query
            endpoint = f"/rest/v1/{table}"
            headers = {"Prefer": "return=representation"}
            
            # Execute the query
            result = await self._make_request(
                method="POST",
                endpoint=endpoint,
                data=data,
                headers=headers,
                auth_token=auth_token,
                use_service_key=auth_token is None
            )
            
            # Invalidate cache for this table
            if redis_service:
                try:
                    # Generate a pattern for related cache keys
                    cache_pattern = f"db:{table}:*"
                    await redis_service.delete_pattern(cache_pattern)
                except Exception as e:
                    logger.error(f"Error invalidating cache after insert: {str(e)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error inserting into database: {str(e)}")
            raise
    
    async def update_db(self, 
                      table: str, 
                      filters: Dict[str, Any],
                      data: Dict[str, Any],
                      auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Update data in the Supabase database
        
        Args:
            table: The table name
            filters: Query filters to identify records to update
            data: The data to update
            auth_token: User auth token
            
        Returns:
            The updated data
            
        Raises:
            SupabaseError: If the update fails
        """
        try:
            # Prepare the query
            endpoint = f"/rest/v1/{table}"
            headers = {"Prefer": "return=representation"}
            
            # Build query parameters from filters
            params = {}
            for key, value in filters.items():
                params[key] = value
            
            # Execute the query
            result = await self._make_request(
                method="PATCH",
                endpoint=endpoint,
                data=data,
                params=params,
                headers=headers,
                auth_token=auth_token,
                use_service_key=auth_token is None
            )
            
            # Invalidate cache for this table
            if redis_service:
                try:
                    # Generate a pattern for related cache keys
                    cache_pattern = f"db:{table}:*"
                    await redis_service.delete_pattern(cache_pattern)
                except Exception as e:
                    logger.error(f"Error invalidating cache after update: {str(e)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error updating database: {str(e)}")
            raise
    
    async def delete_db(self, 
                       table: str, 
                       filters: Dict[str, Any],
                       auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete data from the Supabase database
        
        Args:
            table: The table name
            filters: Query filters to identify records to delete
            auth_token: User auth token
            
        Returns:
            The deleted data
            
        Raises:
            SupabaseError: If the deletion fails
        """
        try:
            # Prepare the query
            endpoint = f"/rest/v1/{table}"
            headers = {"Prefer": "return=representation"}
            
            # Build query parameters from filters
            params = {}
            for key, value in filters.items():
                params[key] = value
            
            # Execute the query
            result = await self._make_request(
                method="DELETE",
                endpoint=endpoint,
                params=params,
                headers=headers,
                auth_token=auth_token,
                use_service_key=auth_token is None
            )
            
            # Invalidate cache for this table
            if redis_service:
                try:
                    # Generate a pattern for related cache keys
                    cache_pattern = f"db:{table}:*"
                    await redis_service.delete_pattern(cache_pattern)
                except Exception as e:
                    logger.error(f"Error invalidating cache after delete: {str(e)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error deleting from database: {str(e)}")
            raise

    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user information
        
        Args:
            user_id: The user ID
            user_data: The user data to update
            
        Returns:
            Updated user information
            
        Raises:
            SupabaseError: If the request fails
        """
        # Validate and sanitize the update data
        allowed_fields = [
            "email", "phone", "password", "email_confirmed",
            "phone_confirmed", "confirmed", "user_metadata", 
            "app_metadata", "banned_until"
        ]
        
        sanitized_data = {k: v for k, v in user_data.items() if k in allowed_fields}
        
        # Add audit trail
        if "app_metadata" not in sanitized_data:
            sanitized_data["app_metadata"] = {}
        
        if "audit" not in sanitized_data["app_metadata"]:
            sanitized_data["app_metadata"]["audit"] = []
        
        sanitized_data["app_metadata"]["audit"].append({
            "action": "update_user",
            "timestamp": int(time.time())
        })
        
        # Update user in Supabase Auth
        endpoint = f"/auth/v1/admin/users/{user_id}"
        
        user = await self._make_request(
            method="PUT",
            endpoint=endpoint,
            data=sanitized_data,
            use_service_key=True
        )
        
        # Invalidate cache
        if redis_service:
            try:
                cache_key = f"user:{user_id}"
                await redis_service.delete(cache_key)
            except Exception as e:
                logger.error(f"Error invalidating user cache: {str(e)}")
        
        return user
        
    async def invalidate_refresh_token(self, refresh_token: str) -> bool:
        """
        Invalidate a refresh token by adding it to Supabase's revoked tokens list
        
        Args:
            refresh_token: The refresh token to invalidate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Revoke the token
            endpoint = "/auth/v1/token/revoke"
            
            await self._make_request(
                method="POST",
                endpoint=endpoint,
                data={"token": refresh_token},
                use_service_key=True
            )
            
            logger.info("Refresh token invalidated successfully")
            return True
            
        except SupabaseError as e:
            logger.error(f"Error invalidating refresh token: {str(e)}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error invalidating refresh token: {str(e)}")
            return False
            
    async def invalidate_all_user_sessions(self, user_id: str) -> bool:
        """
        Invalidate all sessions for a user in Supabase
        
        This is a critical security operation for cases like password change,
        account compromise, or administrative action.
        
        Args:
            user_id: The user ID to invalidate sessions for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use Supabase Admin API to sign out user from all devices
            endpoint = f"/auth/v1/admin/users/{user_id}/sessions"
            
            # Making a DELETE request to this endpoint will remove all sessions
            await self._make_request(
                method="DELETE",
                endpoint=endpoint,
                use_service_key=True
            )
            
            # Mark user as having a security action
            await self._make_request(
                method="PUT",
                endpoint=f"/auth/v1/admin/users/{user_id}",
                data={
                    "app_metadata": {
                        "security_action": {
                            "type": "invalidate_all_sessions",
                            "timestamp": int(time.time())
                        }
                    }
                },
                use_service_key=True
            )
            
            # Clear cache
            if redis_service:
                try:
                    cache_key = f"user:{user_id}"
                    await redis_service.delete(cache_key)
                except Exception as e:
                    logger.error(f"Error invalidating user cache: {str(e)}")
            
            logger.info(f"All sessions invalidated for user {user_id}")
            return True
            
        except SupabaseError as e:
            logger.error(f"Error invalidating sessions for user {user_id}: {str(e)}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error invalidating sessions: {str(e)}")
            return False
            
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all active sessions for a user
        
        Args:
            user_id: The user ID
            
        Returns:
            List of active sessions
            
        Raises:
            SupabaseError: If the request fails
        """
        try:
            # Get user sessions from Supabase Auth
            endpoint = f"/auth/v1/admin/users/{user_id}/sessions"
            
            response = await self._make_request(
                method="GET",
                endpoint=endpoint,
                use_service_key=True
            )
            
            return response.get("sessions", [])
            
        except SupabaseError as e:
            if e.status_code == 404:
                # User not found or no sessions
                logger.warning(f"No sessions found for user {user_id}")
                return []
            else:
                # Re-raise other errors
                raise
                
        except Exception as e:
            logger.error(f"Error getting user sessions: {str(e)}")
            raise
            
    async def create_user(
        self, 
        email: str, 
        password: str, 
        user_metadata: Optional[Dict[str, Any]] = None,
        app_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new user in Supabase Auth
        
        Args:
            email: User email
            password: User password
            user_metadata: Additional user metadata
            app_metadata: Application metadata (admin only)
            
        Returns:
            Created user information
            
        Raises:
            SupabaseError: If the request fails
        """
        # Prepare user data
        user_data = {
            "email": email,
            "password": password,
            "email_confirm": True  # Auto-confirm email for testing
        }
        
        # Add metadata if provided
        if user_metadata:
            user_data["user_metadata"] = user_metadata
            
        if app_metadata:
            user_data["app_metadata"] = app_metadata
        
        # Create user
        endpoint = "/auth/v1/admin/users"
        
        user = await self._make_request(
            method="POST",
            endpoint=endpoint,
            data=user_data,
            use_service_key=True
        )
        
        logger.info(f"User created: {user.get('id')}")
        return user
            
    async def health_check(self) -> bool:
        """
        Check if Supabase is accessible and working
        
        Returns:
            True if healthy, False otherwise
        """
        current_time = time.time()
        
        # Only perform actual check every 30 seconds to avoid unnecessary API calls
        if current_time - self._health_check_timestamp < 30:
            return self._health_status
            
        try:
            # Simple endpoint to check if Supabase is up
            endpoint = "/auth/v1/health"
            
            await self._make_request(
                method="GET",
                endpoint=endpoint,
                use_service_key=True
            )
            
            # Update health check timestamp and status
            self._health_check_timestamp = current_time
            self._health_status = True
            
            return True
            
        except Exception as e:
            logger.error(f"Supabase health check failed: {str(e)}")
            
            # Update health check timestamp and status
            self._health_check_timestamp = current_time
            self._health_status = False
            
            return False
            
    @retry(
        retry=retry_if_exception_type(SupabaseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a JWT token with Supabase Auth
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            # Get auth user from token
            endpoint = "/auth/v1/user"
            
            user = await self._make_request(
                method="GET",
                endpoint=endpoint,
                auth_token=token
            )
            
            return user
            
        except SupabaseError as e:
            if e.status_code in [401, 403]:
                # Invalid token
                logger.warning(f"Invalid token: {str(e)}")
                return None
            else:
                # Re-raise other errors
                raise
                
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return None

    # Add imports for scenario-related models
    from app.models.monte_carlo import (
        ScenarioDefinition,
        SavedSimulation,
        MonteCarloSimulationRequest,
        MonteCarloSimulationResult
    )

    # Scenario management methods
    async def get_scenario(self, scenario_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a scenario by ID
        
        Args:
            scenario_id: The scenario ID
            user_id: Optional user ID for permission checking
            
        Returns:
            Scenario data if found, None otherwise
        """
        # First check cache
        cache_key = f"scenario:{scenario_id}"
        
        if redis_service:
            try:
                cached_scenario = await redis_service.get(cache_key)
                if cached_scenario:
                    logger.debug(f"Cache hit for scenario {scenario_id}")
                    return json.loads(cached_scenario)
            except Exception as e:
                logger.error(f"Error checking scenario cache: {str(e)}")
                # Continue with database query if cache fails
        
        try:
            filters = {"id": f"eq.{scenario_id}"}
            
            # Add user_id filter if provided
            if user_id:
                filters["user_id"] = f"eq.{user_id}"
            
            scenarios = await self.query_db(
                table="scenarios",
                filters=filters,
                limit=1,
                use_cache=False  # Don't use automatic cache since we're managing cache manually
            )
            
            if not scenarios:
                logger.warning(f"Scenario not found: {scenario_id}")
                return None
            
            scenario = scenarios[0]
            
            # Cache the result
            if redis_service:
                try:
                    await redis_service.set(
                        cache_key,
                        json.dumps(scenario),
                        ttl=3600  # 1 hour
                    )
                except Exception as e:
                    logger.error(f"Error caching scenario: {str(e)}")
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error getting scenario: {str(e)}")
            raise
    
    async def list_scenarios(
        self, 
        user_id: Optional[str] = None,
        scenario_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        use_cache: bool = True,
        order_by: str = "created_at.desc"
    ) -> List[Dict[str, Any]]:
        """
        List scenarios with optional filtering
        
        Args:
            user_id: Optional user ID to filter by
            scenario_type: Optional scenario type to filter by
            limit: Maximum number of results
            offset: Offset for pagination
            use_cache: Whether to use Redis cache
            order_by: Order by clause
            
        Returns:
            List of scenarios
        """
        # Generate cache key if using cache
        cache_key = None
        if use_cache and redis_service:
            cache_key = f"scenarios:list:{user_id or 'all'}:{scenario_type or 'all'}:{limit}:{offset}:{order_by}"
            
            try:
                cached_result = await redis_service.get(cache_key)
                if cached_result:
                    logger.debug("Cache hit for scenarios list")
                    return json.loads(cached_result)
            except Exception as e:
                logger.error(f"Error checking scenarios list cache: {str(e)}")
                # Continue with query if cache fails
        
        try:
            filters = {}
            
            # Add user_id filter if provided
            if user_id:
                filters["user_id"] = f"eq.{user_id}"
            
            # Add scenario_type filter if provided
            if scenario_type:
                filters["type"] = f"eq.{scenario_type}"
            
            # Execute the query
            scenarios = await self.query_db(
                table="scenarios",
                select="id,name,description,type,created_at,updated_at,user_id",
                filters=filters,
                order=order_by,
                limit=limit,
                use_cache=False  # Don't use automatic cache since we're managing cache manually
            )
            
            # Cache the result
            if cache_key and redis_service:
                try:
                    await redis_service.set(
                        cache_key,
                        json.dumps(scenarios),
                        ttl=300  # 5 minutes
                    )
                except Exception as e:
                    logger.error(f"Error caching scenarios list: {str(e)}")
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error listing scenarios: {str(e)}")
            raise
    
    async def create_scenario(
        self, 
        scenario: Union[ScenarioDefinition, Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Create a new scenario
        
        Args:
            scenario: The scenario definition (either a Pydantic model or a dictionary)
            user_id: The user ID
            
        Returns:
            The created scenario
        """
        try:
            # Prepare the data
            if hasattr(scenario, 'dict'):
                scenario_dict = scenario.dict()
            else:
                scenario_dict = copy.deepcopy(scenario)
            
            scenario_dict["user_id"] = user_id
            
            # Set creation and update timestamps
            current_time = datetime.now().isoformat()
            scenario_dict["created_at"] = current_time
            scenario_dict["updated_at"] = current_time
            
            # Create the scenario
            result = await self.insert_db(
                table="scenarios",
                data=scenario_dict
            )
            
            logger.info(f"Created scenario {scenario_dict.get('id')} for user {user_id}")
            
            # Invalidate list caches
            if redis_service:
                try:
                    cache_pattern = f"scenarios:list:*"
                    await redis_service.delete_pattern(cache_pattern)
                except Exception as e:
                    logger.error(f"Error invalidating scenarios list cache: {str(e)}")
            
            return result[0] if isinstance(result, list) and result else result
            
        except Exception as e:
            logger.error(f"Error creating scenario: {str(e)}")
            raise
    
    async def update_scenario(
        self, 
        scenario_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update a scenario
        
        Args:
            scenario_id: The scenario ID
            data: Updated scenario data
            user_id: Optional user ID for permission checking
            
        Returns:
            Updated scenario if successful, None if not found
        """
        try:
            # Get the current scenario
            current_scenario = await self.get_scenario(scenario_id, user_id)
            
            if not current_scenario:
                logger.warning(f"Scenario not found for update: {scenario_id}")
                return None
            
            # Prepare the update data
            update_data = {**data, "updated_at": datetime.now().isoformat()}
            
            # Build filters
            filters = {"id": f"eq.{scenario_id}"}
            if user_id:
                filters["user_id"] = f"eq.{user_id}"
            
            # Update the scenario
            result = await self.update_db(
                table="scenarios",
                filters=filters,
                data=update_data
            )
            
            # Invalidate cache
            if redis_service:
                try:
                    # Invalidate specific scenario cache
                    cache_key = f"scenario:{scenario_id}"
                    await redis_service.delete(cache_key)
                    
                    # Invalidate list caches
                    cache_pattern = f"scenarios:list:*"
                    await redis_service.delete_pattern(cache_pattern)
                except Exception as e:
                    logger.error(f"Error invalidating scenario cache: {str(e)}")
            
            if not result:
                return None
                
            return result[0] if isinstance(result, list) and result else result
            
        except Exception as e:
            logger.error(f"Error updating scenario: {str(e)}")
            raise
    
    async def delete_scenario(
        self, 
        scenario_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Delete a scenario
        
        Args:
            scenario_id: The scenario ID
            user_id: Optional user ID for permission checking
            
        Returns:
            True if deleted, False if not found
        """
        try:
            # Build filters
            filters = {"id": f"eq.{scenario_id}"}
            if user_id:
                filters["user_id"] = f"eq.{user_id}"
            
            # Delete the scenario
            result = await self.delete_db(
                table="scenarios",
                filters=filters
            )
            
            # Invalidate cache
            if redis_service:
                try:
                    # Invalidate specific scenario cache
                    cache_key = f"scenario:{scenario_id}"
                    await redis_service.delete(cache_key)
                    
                    # Invalidate list caches
                    cache_pattern = f"scenarios:list:*"
                    await redis_service.delete_pattern(cache_pattern)
                except Exception as e:
                    logger.error(f"Error invalidating scenario cache: {str(e)}")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting scenario: {str(e)}")
            raise
    
    # Simulation management methods
    async def get_simulation(self, simulation_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a simulation by ID
        
        Args:
            simulation_id: The simulation ID
            user_id: Optional user ID for permission checking
            
        Returns:
            Simulation data if found, None otherwise
        """
        # First check cache
        cache_key = f"simulation:{simulation_id}"
        
        if redis_service:
            try:
                cached_simulation = await redis_service.get(cache_key)
                if cached_simulation:
                    logger.debug(f"Cache hit for simulation {simulation_id}")
                    return json.loads(cached_simulation)
            except Exception as e:
                logger.error(f"Error checking simulation cache: {str(e)}")
                # Continue with database query if cache fails
        
        try:
            filters = {"id": f"eq.{simulation_id}"}
            
            # Add user_id filter if provided
            if user_id:
                filters["user_id"] = f"eq.{user_id}"
            
            simulations = await self.query_db(
                table="monte_carlo_simulations",
                filters=filters,
                limit=1,
                use_cache=False  # Don't use automatic cache since we're managing cache manually
            )
            
            if not simulations:
                logger.warning(f"Simulation not found: {simulation_id}")
                return None
            
            simulation = simulations[0]
            
            # Cache the result
            if redis_service:
                try:
                    # Only cache small simulations or metadata to avoid Redis memory issues
                    if "result" in simulation and "detailed_paths" in simulation["result"]:
                        # Create a version without detailed paths for caching
                        cache_simulation = copy.deepcopy(simulation)
                        cache_simulation["result"]["detailed_paths"] = None
                        await redis_service.set(
                            cache_key,
                            json.dumps(cache_simulation),
                            ttl=3600  # 1 hour
                        )
                    else:
                        await redis_service.set(
                            cache_key,
                            json.dumps(simulation),
                            ttl=3600  # 1 hour
                        )
                except Exception as e:
                    logger.error(f"Error caching simulation: {str(e)}")
            
            return simulation
            
        except Exception as e:
            logger.error(f"Error getting simulation: {str(e)}")
            raise
    
    async def list_simulations(
        self, 
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        use_cache: bool = True,
        order_by: str = "created_at.desc"
    ) -> List[Dict[str, Any]]:
        """
        List simulations with optional filtering
        
        Args:
            user_id: Optional user ID to filter by
            status: Optional status to filter by
            limit: Maximum number of results
            offset: Offset for pagination
            use_cache: Whether to use Redis cache
            order_by: Order by clause
            
        Returns:
            List of simulations
        """
        # Generate cache key if using cache
        cache_key = None
        if use_cache and redis_service:
            cache_key = f"simulations:list:{user_id or 'all'}:{status or 'all'}:{limit}:{offset}:{order_by}"
            
            try:
                cached_result = await redis_service.get(cache_key)
                if cached_result:
                    logger.debug("Cache hit for simulations list")
                    return json.loads(cached_result)
            except Exception as e:
                logger.error(f"Error checking simulations list cache: {str(e)}")
                # Continue with query if cache fails
        
        try:
            filters = {}
            
            # Add user_id filter if provided
            if user_id:
                filters["user_id"] = f"eq.{user_id}"
            
            # Using JSON operators to filter by status
            if status:
                filters["result->status"] = f"eq.{status}"
            
            # Execute the query, selecting only necessary fields
            select_fields = "id,name,description,user_id,created_at,updated_at,result->>status as status"
            
            simulations = await self.query_db(
                table="monte_carlo_simulations",
                select=select_fields,
                filters=filters,
                order=order_by,
                limit=limit,
                use_cache=False  # Don't use automatic cache since we're managing cache manually
            )
            
            # Cache the result
            if cache_key and redis_service:
                try:
                    await redis_service.set(
                        cache_key,
                        json.dumps(simulations),
                        ttl=300  # 5 minutes
                    )
                except Exception as e:
                    logger.error(f"Error caching simulations list: {str(e)}")
            
            return simulations
            
        except Exception as e:
            logger.error(f"Error listing simulations: {str(e)}")
            raise
    
    async def create_simulation(
        self, 
        simulation: SavedSimulation
    ) -> Dict[str, Any]:
        """
        Create a new simulation record
        
        Args:
            simulation: The simulation to create
            
        Returns:
            The created simulation
        """
        try:
            # Prepare the data
            simulation_dict = simulation.dict()
            
            # Set creation and update timestamps if not already set
            current_time = datetime.now().isoformat()
            if "created_at" not in simulation_dict:
                simulation_dict["created_at"] = current_time
            if "updated_at" not in simulation_dict:
                simulation_dict["updated_at"] = current_time
            
            # Create the simulation
            result = await self.insert_db(
                table="monte_carlo_simulations",
                data=simulation_dict
            )
            
            logger.info(f"Created simulation {simulation.id} for user {simulation.user_id}")
            
            # Invalidate list caches
            if redis_service:
                try:
                    cache_pattern = f"simulations:list:*"
                    await redis_service.delete_pattern(cache_pattern)
                except Exception as e:
                    logger.error(f"Error invalidating simulations list cache: {str(e)}")
            
            return result[0] if isinstance(result, list) and result else result
            
        except Exception as e:
            logger.error(f"Error creating simulation: {str(e)}")
            raise
    
    async def update_simulation(
        self, 
        simulation_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update a simulation
        
        Args:
            simulation_id: The simulation ID
            data: Updated simulation data
            user_id: Optional user ID for permission checking
            
        Returns:
            Updated simulation if successful, None if not found
        """
        try:
            # Prepare the update data
            update_data = {**data, "updated_at": datetime.now().isoformat()}
            
            # Build filters
            filters = {"id": f"eq.{simulation_id}"}
            if user_id:
                filters["user_id"] = f"eq.{user_id}"
            
            # Update the simulation
            result = await self.update_db(
                table="monte_carlo_simulations",
                filters=filters,
                data=update_data
            )
            
            # Invalidate cache
            if redis_service:
                try:
                    # Invalidate specific simulation cache
                    cache_key = f"simulation:{simulation_id}"
                    await redis_service.delete(cache_key)
                    
                    # Invalidate list caches
                    cache_pattern = f"simulations:list:*"
                    await redis_service.delete_pattern(cache_pattern)
                except Exception as e:
                    logger.error(f"Error invalidating simulation cache: {str(e)}")
            
            if not result:
                return None
                
            return result[0] if isinstance(result, list) and result else result
            
        except Exception as e:
            logger.error(f"Error updating simulation: {str(e)}")
            raise
    
    # Synchronous methods for worker tasks
    def get_item_sync(self, table: str, id: str) -> Optional[Dict[str, Any]]:
        """
        Get an item by ID synchronously (for worker tasks)
        
        Args:
            table: The table name
            id: The item ID
            
        Returns:
            Item data if found, None otherwise
        """
        try:
            # Create synchronous HTTP client
            with httpx.Client(timeout=30.0) as client:
                # Prepare the request
                url = f"{self.base_url}/rest/v1/{table}"
                params = {"id": f"eq.{id}", "limit": "1"}
                
                headers = {
                    "apikey": self.anon_key,
                    "Authorization": f"Bearer {self.service_key}"
                }
                
                # Make the request
                response = client.get(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    items = response.json()
                    return items[0] if items else None
                elif response.status_code == 404:
                    return None
                else:
                    response.raise_for_status()
                    
        except httpx.RequestError as e:
            logger.error(f"Error getting item from {table}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting item from {table}: {str(e)}")
            return None
    
    def update_item_sync(self, table: str, id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an item by ID synchronously (for worker tasks)
        
        Args:
            table: The table name
            id: The item ID
            data: The data to update
            
        Returns:
            Updated item if successful, None otherwise
        """
        try:
            # Create synchronous HTTP client
            with httpx.Client(timeout=30.0) as client:
                # Prepare the request
                url = f"{self.base_url}/rest/v1/{table}"
                params = {"id": f"eq.{id}"}
                
                headers = {
                    "apikey": self.anon_key,
                    "Authorization": f"Bearer {self.service_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
                
                # Make the request
                response = client.patch(url, params=params, headers=headers, json=data)
                
                if response.status_code == 200:
                    items = response.json()
                    return items[0] if items else None
                elif response.status_code == 404:
                    return None
                else:
                    response.raise_for_status()
                    
        except httpx.RequestError as e:
            logger.error(f"Error updating item in {table}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error updating item in {table}: {str(e)}")
            return None
    
    def query_items_sync(self, table: str, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query items synchronously (for worker tasks)
        
        Args:
            table: The table name
            query: Query parameters
            limit: Maximum number of results
            
        Returns:
            List of items
        """
        try:
            # Create synchronous HTTP client
            with httpx.Client(timeout=30.0) as client:
                # Prepare the request
                url = f"{self.base_url}/rest/v1/{table}"
                
                # Build query parameters
                params = {**query, "limit": str(limit)}
                
                headers = {
                    "apikey": self.anon_key,
                    "Authorization": f"Bearer {self.service_key}"
                }
                
                # Make the request
                response = client.get(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    response.raise_for_status()
                    
        except httpx.RequestError as e:
            logger.error(f"Error querying items from {table}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error querying items from {table}: {str(e)}")
            return []
    
    def create_item_sync(self, table: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create an item synchronously (for worker tasks)
        
        Args:
            table: The table name
            data: The item data
            
        Returns:
            Created item if successful, None otherwise
        """
        try:
            # Create synchronous HTTP client
            with httpx.Client(timeout=30.0) as client:
                # Prepare the request
                url = f"{self.base_url}/rest/v1/{table}"
                
                headers = {
                    "apikey": self.anon_key,
                    "Authorization": f"Bearer {self.service_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
                
                # Make the request
                response = client.post(url, headers=headers, json=data)
                
                if response.status_code in [200, 201]:
                    items = response.json()
                    return items[0] if items else None
                else:
                    response.raise_for_status()
                    
        except httpx.RequestError as e:
            logger.error(f"Error creating item in {table}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating item in {table}: {str(e)}")
            return None

"""
Deal Library Service for AbsBox Integration

This service implements the deal library functionality as described in the AbsBox documentation:
https://absbox-doc.readthedocs.io/en/latest/library.html

It provides:
1. A public deal repository accessible to all users
2. Private deals accessible only to their owners
3. Shared private deals accessible to users with explicit permissions
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import json
import os
from datetime import datetime
from uuid import UUID, uuid4

# Database imports
from sqlalchemy import and_, or_
from app.db import models, schemas
from app.db.session import SessionLocal
from app.db.crud import deal as deal_crud, user as user_crud

# AbsBox imports
import absbox as ab

# Core services
from app.core.cache_service import CacheService
from app.core.security import get_current_user_id
from app.core.error_handling import handle_exceptions
from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

class DealLibraryService:
    """
    Deal Library Service for structured finance analytics using AbsBox
    
    This service manages the deal library functionality, providing:
    - Public deal repository
    - Private user deals
    - Deal sharing capabilities
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        """
        Initialize the Deal Library service
        
        Args:
            cache_service: Optional cache service for improved performance
        """
        self.cache = cache_service or CacheService()
        
        # AbsBox integration
        try:
            # Access AbsBox library functionality
            from absbox.library import DealLibrary
            self.deal_library = DealLibrary()
            self.library_available = True
            logger.info("AbsBox Deal Library initialized successfully")
        except (ImportError, AttributeError) as e:
            logger.warning(f"AbsBox Deal Library not available: {e}")
            self.library_available = False
            self.deal_library = None
            
        logger.info("Deal Library Service initialized")

    @handle_exceptions
    async def get_public_deals(self, 
                          skip: int = 0, 
                          limit: int = 100,
                          filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get list of public deals available to all users
        
        Args:
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of public deal metadata
        """
        cache_key = f"public_deals:{skip}:{limit}:{json.dumps(filter_criteria or {})}"
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
            
        # Fetch from database
        async with SessionLocal() as db:
            deals = await deal_crud.get_public_deals(
                db=db,
                skip=skip,
                limit=limit,
                filter_criteria=filter_criteria
            )
            
            result = [deal.dict() for deal in deals]
            
            # Cache the results
            await self.cache.set(cache_key, json.dumps(result), expire=3600)  # Cache for 1 hour
            
            return result
    
    @handle_exceptions
    async def get_user_deals(self,
                        user_id: UUID,
                        skip: int = 0,
                        limit: int = 100,
                        filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get list of deals owned by a specific user
        
        Args:
            user_id: ID of the user
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of user's deal metadata
        """
        cache_key = f"user_deals:{user_id}:{skip}:{limit}:{json.dumps(filter_criteria or {})}"
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
            
        # Fetch from database
        async with SessionLocal() as db:
            deals = await deal_crud.get_user_deals(
                db=db,
                user_id=user_id,
                skip=skip,
                limit=limit,
                filter_criteria=filter_criteria
            )
            
            result = [deal.dict() for deal in deals]
            
            # Cache the results
            await self.cache.set(cache_key, json.dumps(result), expire=3600)  # Cache for 1 hour
            
            return result
    
    @handle_exceptions
    async def get_shared_deals(self,
                          user_id: UUID,
                          skip: int = 0,
                          limit: int = 100,
                          filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get list of deals shared with a specific user
        
        Args:
            user_id: ID of the user
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of shared deal metadata
        """
        cache_key = f"shared_deals:{user_id}:{skip}:{limit}:{json.dumps(filter_criteria or {})}"
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
            
        # Fetch from database
        async with SessionLocal() as db:
            deals = await deal_crud.get_shared_deals(
                db=db,
                user_id=user_id,
                skip=skip,
                limit=limit,
                filter_criteria=filter_criteria
            )
            
            result = [deal.dict() for deal in deals]
            
            # Cache the results
            await self.cache.set(cache_key, json.dumps(result), expire=3600)  # Cache for 1 hour
            
            return result
    
    @handle_exceptions
    async def get_deal_by_id(self, deal_id: UUID, user_id: Optional[UUID] = None) -> Dict[str, Any]:
        """
        Get a specific deal by ID, checking access permissions
        
        Args:
            deal_id: ID of the deal to retrieve
            user_id: Optional user ID for permission checking
            
        Returns:
            Deal data if accessible
            
        Raises:
            PermissionError: If user doesn't have access to the deal
            ValueError: If deal doesn't exist
        """
        cache_key = f"deal:{deal_id}:{user_id or 'public'}"
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
            
        # Fetch from database with permission check
        async with SessionLocal() as db:
            deal = await deal_crud.get_deal_by_id(db=db, deal_id=deal_id)
            
            if not deal:
                raise ValueError(f"Deal with ID {deal_id} not found")
                
            # Check permissions
            if not deal.is_public:
                if not user_id:
                    raise PermissionError("Authentication required for private deals")
                    
                # Check if user owns the deal
                if deal.owner_id != user_id:
                    # Check if deal is shared with this user
                    shared_access = await deal_crud.check_deal_access(
                        db=db,
                        deal_id=deal_id,
                        user_id=user_id
                    )
                    
                    if not shared_access:
                        raise PermissionError("You don't have access to this deal")
            
            result = deal.dict()
            
            # Cache the results
            await self.cache.set(cache_key, json.dumps(result), expire=3600)  # Cache for 1 hour
            
            return result
    
    @handle_exceptions
    async def create_deal(self, 
                     deal_data: Dict[str, Any], 
                     user_id: UUID,
                     is_public: bool = False) -> Dict[str, Any]:
        """
        Create a new deal in the library
        
        Args:
            deal_data: Deal configuration data
            user_id: ID of the user creating the deal
            is_public: Whether the deal should be public
            
        Returns:
            Created deal metadata
        """
        # Validate deal data structure
        if not deal_data.get("name"):
            raise ValueError("Deal name is required")
            
        # Create in database
        async with SessionLocal() as db:
            deal_create = schemas.DealCreate(
                name=deal_data.get("name"),
                description=deal_data.get("description", ""),
                deal_type=deal_data.get("deal_type", "custom"),
                structure=deal_data,
                is_public=is_public,
                owner_id=user_id
            )
            
            new_deal = await deal_crud.create_deal(db=db, deal=deal_create)
            
            # Clear relevant caches
            await self._invalidate_deal_caches(user_id)
            
            return new_deal.dict()
    
    @handle_exceptions
    async def update_deal(self, 
                     deal_id: UUID, 
                     deal_data: Dict[str, Any], 
                     user_id: UUID) -> Dict[str, Any]:
        """
        Update an existing deal
        
        Args:
            deal_id: ID of the deal to update
            deal_data: Updated deal configuration
            user_id: ID of the user making the update
            
        Returns:
            Updated deal metadata
            
        Raises:
            PermissionError: If user doesn't have permission to update
            ValueError: If deal doesn't exist
        """
        # Check deal existence and ownership
        async with SessionLocal() as db:
            existing_deal = await deal_crud.get_deal_by_id(db=db, deal_id=deal_id)
            
            if not existing_deal:
                raise ValueError(f"Deal with ID {deal_id} not found")
                
            # Only the owner can update a deal
            if existing_deal.owner_id != user_id:
                # Special case: admin override (can be customized as needed)
                if await user_crud.is_admin(db=db, user_id=user_id):
                    logger.warning(f"Admin {user_id} updating deal {deal_id} owned by {existing_deal.owner_id}")
                else:
                    raise PermissionError("Only the owner can update this deal")
            
            # Update the deal
            deal_update = schemas.DealUpdate(
                name=deal_data.get("name", existing_deal.name),
                description=deal_data.get("description", existing_deal.description),
                deal_type=deal_data.get("deal_type", existing_deal.deal_type),
                structure=deal_data.get("structure", existing_deal.structure),
                is_public=deal_data.get("is_public", existing_deal.is_public)
            )
            
            updated_deal = await deal_crud.update_deal(
                db=db,
                deal_id=deal_id,
                deal_update=deal_update
            )
            
            # Clear relevant caches
            await self._invalidate_deal_caches(user_id)
            if existing_deal.is_public or deal_update.is_public:
                await self._invalidate_public_deal_caches()
                
            return updated_deal.dict()
    
    @handle_exceptions
    async def delete_deal(self, deal_id: UUID, user_id: UUID) -> bool:
        """
        Delete a deal from the library
        
        Args:
            deal_id: ID of the deal to delete
            user_id: ID of the user making the deletion
            
        Returns:
            True if deleted successfully
            
        Raises:
            PermissionError: If user doesn't have permission to delete
            ValueError: If deal doesn't exist
        """
        # Check deal existence and ownership
        async with SessionLocal() as db:
            existing_deal = await deal_crud.get_deal_by_id(db=db, deal_id=deal_id)
            
            if not existing_deal:
                raise ValueError(f"Deal with ID {deal_id} not found")
                
            # Only the owner can delete a deal
            if existing_deal.owner_id != user_id:
                # Special case: admin override
                if await user_crud.is_admin(db=db, user_id=user_id):
                    logger.warning(f"Admin {user_id} deleting deal {deal_id} owned by {existing_deal.owner_id}")
                else:
                    raise PermissionError("Only the owner can delete this deal")
            
            # Delete the deal
            success = await deal_crud.delete_deal(db=db, deal_id=deal_id)
            
            if success:
                # Clear relevant caches
                await self._invalidate_deal_caches(user_id)
                if existing_deal.is_public:
                    await self._invalidate_public_deal_caches()
                
                # Remove any shared access entries
                await deal_crud.remove_all_deal_access(db=db, deal_id=deal_id)
                
            return success
    
    @handle_exceptions
    async def share_deal(self, 
                    deal_id: UUID, 
                    owner_id: UUID, 
                    shared_with_id: UUID,
                    permission_level: str = "read") -> bool:
        """
        Share a deal with another user
        
        Args:
            deal_id: ID of the deal to share
            owner_id: ID of the deal owner
            shared_with_id: ID of the user to share with
            permission_level: Permission level (read, edit, admin)
            
        Returns:
            True if shared successfully
            
        Raises:
            PermissionError: If user doesn't have permission to share
            ValueError: If deal doesn't exist
        """
        # Check deal existence and ownership
        async with SessionLocal() as db:
            existing_deal = await deal_crud.get_deal_by_id(db=db, deal_id=deal_id)
            
            if not existing_deal:
                raise ValueError(f"Deal with ID {deal_id} not found")
                
            # Only the owner can share a deal
            if existing_deal.owner_id != owner_id:
                # Special case: admin override
                if await user_crud.is_admin(db=db, user_id=owner_id):
                    logger.warning(f"Admin {owner_id} sharing deal {deal_id} owned by {existing_deal.owner_id}")
                else:
                    raise PermissionError("Only the owner can share this deal")
            
            # Check if target user exists
            target_user = await user_crud.get_user(db=db, user_id=shared_with_id)
            if not target_user:
                raise ValueError(f"User with ID {shared_with_id} not found")
                
            # Create deal access entry
            access_create = schemas.DealAccessCreate(
                deal_id=deal_id,
                user_id=shared_with_id,
                permission_level=permission_level
            )
            
            access = await deal_crud.create_deal_access(db=db, access=access_create)
            
            # Clear relevant caches
            await self._invalidate_deal_caches(shared_with_id)
                
            return bool(access)
    
    @handle_exceptions
    async def revoke_deal_access(self, 
                            deal_id: UUID, 
                            owner_id: UUID, 
                            shared_with_id: UUID) -> bool:
        """
        Revoke a user's access to a shared deal
        
        Args:
            deal_id: ID of the deal
            owner_id: ID of the deal owner
            shared_with_id: ID of the user to revoke access from
            
        Returns:
            True if access revoked successfully
            
        Raises:
            PermissionError: If user doesn't have permission to revoke
            ValueError: If deal doesn't exist
        """
        # Check deal existence and ownership
        async with SessionLocal() as db:
            existing_deal = await deal_crud.get_deal_by_id(db=db, deal_id=deal_id)
            
            if not existing_deal:
                raise ValueError(f"Deal with ID {deal_id} not found")
                
            # Only the owner can revoke access
            if existing_deal.owner_id != owner_id:
                # Special case: admin override
                if await user_crud.is_admin(db=db, user_id=owner_id):
                    logger.warning(f"Admin {owner_id} revoking access for deal {deal_id} owned by {existing_deal.owner_id}")
                else:
                    raise PermissionError("Only the owner can revoke access")
            
            # Remove deal access
            success = await deal_crud.remove_deal_access(
                db=db,
                deal_id=deal_id,
                user_id=shared_with_id
            )
            
            if success:
                # Clear relevant caches
                await self._invalidate_deal_caches(shared_with_id)
                
            return success
    
    @handle_exceptions
    async def get_deal_access_list(self, 
                              deal_id: UUID, 
                              owner_id: UUID) -> List[Dict[str, Any]]:
        """
        Get list of users with access to a deal
        
        Args:
            deal_id: ID of the deal
            owner_id: ID of the deal owner
            
        Returns:
            List of users with access and their permission levels
            
        Raises:
            PermissionError: If user doesn't have permission to view
            ValueError: If deal doesn't exist
        """
        # Check deal existence and ownership
        async with SessionLocal() as db:
            existing_deal = await deal_crud.get_deal_by_id(db=db, deal_id=deal_id)
            
            if not existing_deal:
                raise ValueError(f"Deal with ID {deal_id} not found")
                
            # Only the owner can view access list
            if existing_deal.owner_id != owner_id:
                # Special case: admin override
                if await user_crud.is_admin(db=db, user_id=owner_id):
                    logger.warning(f"Admin {owner_id} viewing access list for deal {deal_id} owned by {existing_deal.owner_id}")
                else:
                    raise PermissionError("Only the owner can view the access list")
            
            # Get access list
            access_list = await deal_crud.get_deal_access_list(db=db, deal_id=deal_id)
            
            # Enrich with user data
            result = []
            for access in access_list:
                user = await user_crud.get_user(db=db, user_id=access.user_id)
                if user:
                    result.append({
                        "user_id": str(access.user_id),
                        "username": user.username,
                        "email": user.email,
                        "permission_level": access.permission_level,
                        "shared_at": access.created_at.isoformat()
                    })
                
            return result
    
    @handle_exceptions
    async def export_deal_to_absbox(self, deal_id: UUID, user_id: Optional[UUID] = None) -> Dict[str, Any]:
        """
        Export a deal to AbsBox format for analysis
        
        Args:
            deal_id: ID of the deal to export
            user_id: Optional user ID for permission checking
            
        Returns:
            Deal in AbsBox format
            
        Raises:
            PermissionError: If user doesn't have access to the deal
            ValueError: If deal doesn't exist
        """
        # Get the deal with permission check
        deal_data = await self.get_deal_by_id(deal_id=deal_id, user_id=user_id)
        
        # Convert to AbsBox format
        structure = deal_data.get("structure", {})
        
        # Process structure to ensure compatibility with AbsBox
        # This may involve transforming certain fields or adding required attributes
        
        # If AbsBox library available, validate with its API
        if self.library_available and self.deal_library:
            try:
                # Use AbsBox validation if available
                self.deal_library.validate_deal(structure)
            except Exception as e:
                logger.error(f"AbsBox validation error: {e}")
                raise ValueError(f"Deal structure is not compatible with AbsBox: {e}")
        
        return structure
    
    @handle_exceptions
    async def import_deal_from_absbox(self, 
                                 deal_data: Dict[str, Any], 
                                 user_id: UUID,
                                 name: str,
                                 description: str = "",
                                 is_public: bool = False) -> Dict[str, Any]:
        """
        Import a deal from AbsBox format
        
        Args:
            deal_data: Deal in AbsBox format
            user_id: ID of the user importing the deal
            name: Name for the imported deal
            description: Optional description
            is_public: Whether the deal should be public
            
        Returns:
            Imported deal metadata
        """
        # Validate with AbsBox if available
        if self.library_available and self.deal_library:
            try:
                # Use AbsBox validation if available
                self.deal_library.validate_deal(deal_data)
            except Exception as e:
                logger.error(f"AbsBox validation error: {e}")
                raise ValueError(f"Deal structure is not compatible with AbsBox: {e}")
        
        # Create the deal
        return await self.create_deal(
            deal_data={
                "name": name,
                "description": description,
                "deal_type": "imported",
                "structure": deal_data
            },
            user_id=user_id,
            is_public=is_public
        )
    
    @handle_exceptions
    async def clone_deal(self, 
                    deal_id: UUID, 
                    user_id: UUID,
                    new_name: Optional[str] = None,
                    new_description: Optional[str] = None,
                    is_public: bool = False) -> Dict[str, Any]:
        """
        Clone an existing deal
        
        Args:
            deal_id: ID of the deal to clone
            user_id: ID of the user cloning the deal
            new_name: Optional new name for the cloned deal
            new_description: Optional new description
            is_public: Whether the cloned deal should be public
            
        Returns:
            Cloned deal metadata
        """
        # Get the deal with permission check
        deal_data = await self.get_deal_by_id(deal_id=deal_id, user_id=user_id)
        
        # Create a new deal with the same structure
        structure = deal_data.get("structure", {})
        
        return await self.create_deal(
            deal_data={
                "name": new_name or f"Clone of {deal_data.get('name')}",
                "description": new_description or f"Cloned from {deal_data.get('name')}",
                "deal_type": deal_data.get("deal_type", "custom"),
                "structure": structure
            },
            user_id=user_id,
            is_public=is_public
        )
    
    # Private helper methods
    
    async def _invalidate_deal_caches(self, user_id: UUID) -> None:
        """Clear caches related to a user's deals"""
        user_id_str = str(user_id)
        await self.cache.delete_pattern(f"user_deals:{user_id_str}:*")
        await self.cache.delete_pattern(f"shared_deals:{user_id_str}:*")
        await self.cache.delete_pattern(f"deal:*:{user_id_str}")
    
    async def _invalidate_public_deal_caches(self) -> None:
        """Clear caches related to public deals"""
        await self.cache.delete_pattern("public_deals:*")
        await self.cache.delete_pattern("deal:*:public")

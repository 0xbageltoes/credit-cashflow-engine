"""
API endpoints for the AbsBox Deal Library.

These endpoints allow the frontend to interact with the deal library functionality:
1. Browse and search public deals
2. Manage personal deals (create, update, delete)
3. Share deals with other users
4. Import/export deals to/from AbsBox format
"""
from typing import List, Dict, Optional, Any, Union
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status, Path, Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.services.deal_library_service import DealLibraryService
from app.db import schemas
from app.core.security import get_current_user_id

router = APIRouter()

# Initialize the deal library service with caching
deal_library_service = None


async def get_deal_library_service(
    db: AsyncSession = Depends(deps.get_db),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """Dependency to get deal library service with proper initialization"""
    global deal_library_service
    if deal_library_service is None:
        # Initialize with cache service from deps
        from app.core.cache_service import get_cache_service
        cache_service = await get_cache_service()
        deal_library_service = DealLibraryService(cache_service=cache_service)
    return deal_library_service


@router.get("/public", response_model=List[schemas.deal.DealSummary])
async def list_public_deals(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Maximum number of records to return"),
    name: Optional[str] = Query(None, description="Filter by name (partial match)"),
    deal_type: Optional[str] = Query(None, description="Filter by deal type"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user_optional)
):
    """
    List all public deals available in the library.
    
    This endpoint returns a paginated list of public deals that can be accessed by all users.
    Optional filters can be applied to search by name or deal type.
    """
    filter_criteria = {}
    if name:
        filter_criteria["name"] = name
    if deal_type:
        filter_criteria["deal_type"] = deal_type
    
    return await service.get_public_deals(
        skip=skip,
        limit=limit,
        filter_criteria=filter_criteria
    )


@router.get("/my-deals", response_model=List[schemas.deal.DealSummary])
async def list_my_deals(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Maximum number of records to return"),
    name: Optional[str] = Query(None, description="Filter by name (partial match)"),
    deal_type: Optional[str] = Query(None, description="Filter by deal type"),
    is_public: Optional[bool] = Query(None, description="Filter by public status"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    List all deals owned by the current user.
    
    This endpoint returns a paginated list of deals owned by the currently authenticated user.
    Optional filters can be applied to search by name, deal type, or public status.
    """
    user_id = UUID(str(current_user.id))
    
    filter_criteria = {}
    if name:
        filter_criteria["name"] = name
    if deal_type:
        filter_criteria["deal_type"] = deal_type
    if is_public is not None:
        filter_criteria["is_public"] = is_public
    
    return await service.get_user_deals(
        user_id=user_id,
        skip=skip,
        limit=limit,
        filter_criteria=filter_criteria
    )


@router.get("/shared-with-me", response_model=List[schemas.deal.DealSummary])
async def list_shared_deals(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Maximum number of records to return"),
    name: Optional[str] = Query(None, description="Filter by name (partial match)"),
    deal_type: Optional[str] = Query(None, description="Filter by deal type"),
    permission_level: Optional[str] = Query(None, description="Filter by permission level"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    List all deals shared with the current user.
    
    This endpoint returns a paginated list of deals that have been shared with the currently
    authenticated user by other users. Optional filters can be applied to search by name, 
    deal type, or permission level.
    """
    user_id = UUID(str(current_user.id))
    
    filter_criteria = {}
    if name:
        filter_criteria["name"] = name
    if deal_type:
        filter_criteria["deal_type"] = deal_type
    if permission_level:
        filter_criteria["permission_level"] = permission_level
    
    return await service.get_shared_deals(
        user_id=user_id,
        skip=skip,
        limit=limit,
        filter_criteria=filter_criteria
    )


@router.get("/{deal_id}", response_model=schemas.deal.Deal)
async def get_deal(
    deal_id: UUID = Path(..., description="ID of the deal to retrieve"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: Optional[schemas.User] = Depends(deps.get_current_user_optional)
):
    """
    Get details of a specific deal.
    
    This endpoint retrieves the complete details of a deal, including its structure.
    If the deal is private, the user must be authenticated and have proper access.
    """
    try:
        user_id = UUID(str(current_user.id)) if current_user else None
        return await service.get_deal_by_id(deal_id=deal_id, user_id=user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.post("/", response_model=schemas.deal.Deal, status_code=status.HTTP_201_CREATED)
async def create_deal(
    deal_data: Dict[str, Any] = Body(..., description="Deal configuration data"),
    is_public: bool = Query(False, description="Whether the deal should be public"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Create a new deal.
    
    This endpoint creates a new deal in the library with the provided configuration.
    The current user will be set as the owner of the deal.
    """
    try:
        user_id = UUID(str(current_user.id))
        return await service.create_deal(
            deal_data=deal_data,
            user_id=user_id,
            is_public=is_public
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/{deal_id}", response_model=schemas.deal.Deal)
async def update_deal(
    deal_id: UUID = Path(..., description="ID of the deal to update"),
    deal_data: Dict[str, Any] = Body(..., description="Updated deal configuration"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Update an existing deal.
    
    This endpoint updates an existing deal with the provided configuration.
    The user must be the owner of the deal to update it.
    """
    try:
        user_id = UUID(str(current_user.id))
        return await service.update_deal(
            deal_id=deal_id,
            deal_data=deal_data,
            user_id=user_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.delete("/{deal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deal(
    deal_id: UUID = Path(..., description="ID of the deal to delete"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Delete a deal.
    
    This endpoint deletes a deal from the library.
    The user must be the owner of the deal to delete it.
    """
    try:
        user_id = UUID(str(current_user.id))
        success = await service.delete_deal(
            deal_id=deal_id,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deal with ID {deal_id} not found"
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.post("/{deal_id}/share", response_model=bool)
async def share_deal(
    deal_id: UUID = Path(..., description="ID of the deal to share"),
    shared_with_id: UUID = Body(..., description="ID of the user to share with"),
    permission_level: str = Body("read", description="Permission level (read, edit, admin)"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Share a deal with another user.
    
    This endpoint grants access to a deal for another user with the specified permission level.
    The current user must be the owner of the deal to share it.
    """
    try:
        user_id = UUID(str(current_user.id))
        return await service.share_deal(
            deal_id=deal_id,
            owner_id=user_id,
            shared_with_id=shared_with_id,
            permission_level=permission_level
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.delete("/{deal_id}/share/{user_id}", response_model=bool)
async def revoke_deal_access(
    deal_id: UUID = Path(..., description="ID of the deal"),
    user_id: UUID = Path(..., description="ID of the user to revoke access from"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Revoke a user's access to a shared deal.
    
    This endpoint removes access to a deal for the specified user.
    The current user must be the owner of the deal to revoke access.
    """
    try:
        owner_id = UUID(str(current_user.id))
        return await service.revoke_deal_access(
            deal_id=deal_id,
            owner_id=owner_id,
            shared_with_id=user_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.get("/{deal_id}/access", response_model=List[Dict[str, Any]])
async def get_deal_access_list(
    deal_id: UUID = Path(..., description="ID of the deal"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Get list of users with access to a deal.
    
    This endpoint returns a list of users who have been granted access to the deal,
    along with their permission levels. The current user must be the owner of the deal.
    """
    try:
        owner_id = UUID(str(current_user.id))
        return await service.get_deal_access_list(
            deal_id=deal_id,
            owner_id=owner_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.post("/{deal_id}/export", response_model=Dict[str, Any])
async def export_deal_to_absbox(
    deal_id: UUID = Path(..., description="ID of the deal to export"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: Optional[schemas.User] = Depends(deps.get_current_user_optional)
):
    """
    Export a deal to AbsBox format.
    
    This endpoint converts a deal to AbsBox-compatible format for analysis.
    If the deal is private, the user must be authenticated and have proper access.
    """
    try:
        user_id = UUID(str(current_user.id)) if current_user else None
        return await service.export_deal_to_absbox(
            deal_id=deal_id,
            user_id=user_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.post("/import", response_model=schemas.deal.Deal, status_code=status.HTTP_201_CREATED)
async def import_deal_from_absbox(
    deal_data: Dict[str, Any] = Body(..., description="Deal in AbsBox format"),
    name: str = Body(..., description="Name for the imported deal"),
    description: Optional[str] = Body("", description="Description for the imported deal"),
    is_public: bool = Query(False, description="Whether the deal should be public"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Import a deal from AbsBox format.
    
    This endpoint creates a new deal in the library from AbsBox-compatible data.
    The current user will be set as the owner of the imported deal.
    """
    try:
        user_id = UUID(str(current_user.id))
        return await service.import_deal_from_absbox(
            deal_data=deal_data,
            user_id=user_id,
            name=name,
            description=description,
            is_public=is_public
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/{deal_id}/clone", response_model=schemas.deal.Deal, status_code=status.HTTP_201_CREATED)
async def clone_deal(
    deal_id: UUID = Path(..., description="ID of the deal to clone"),
    new_name: Optional[str] = Query(None, description="Optional name for the cloned deal"),
    new_description: Optional[str] = Query(None, description="Optional description for the cloned deal"),
    is_public: bool = Query(False, description="Whether the cloned deal should be public"),
    service: DealLibraryService = Depends(get_deal_library_service),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    """
    Clone an existing deal.
    
    This endpoint creates a new deal as a copy of an existing one.
    The current user will be set as the owner of the cloned deal.
    The user must have access to the original deal to clone it.
    """
    try:
        user_id = UUID(str(current_user.id))
        return await service.clone_deal(
            deal_id=deal_id,
            user_id=user_id,
            new_name=new_name,
            new_description=new_description,
            is_public=is_public
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )

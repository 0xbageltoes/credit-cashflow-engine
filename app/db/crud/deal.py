"""
CRUD operations for the AbsBox Deal Library.

This module provides database operations for:
1. Deal management (create, read, update, delete)
2. Deal access control (permissions, sharing)
3. Deal versioning (history tracking)
"""
from typing import List, Dict, Optional, Any, Union, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from fastapi.encoders import jsonable_encoder

from app.db.models.deal import Deal, DealAccess, DealVersion
from app.db.schemas import deal as schemas


async def get_deals(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> List[Deal]:
    """
    Get a list of deals with pagination and optional filtering
    
    Args:
        db: Database session
        skip: Records to skip (pagination)
        limit: Maximum records to return
        filter_criteria: Optional filtering criteria
        
    Returns:
        List of Deal objects
    """
    query = select(Deal).where(Deal.is_deleted == False)
    
    # Apply filters if provided
    if filter_criteria:
        if "name" in filter_criteria:
            query = query.where(Deal.name.ilike(f"%{filter_criteria['name']}%"))
        
        if "deal_type" in filter_criteria:
            query = query.where(Deal.deal_type == filter_criteria["deal_type"])
        
        if "owner_id" in filter_criteria:
            query = query.where(Deal.owner_id == filter_criteria["owner_id"])
    
    # Apply pagination
    query = query.offset(skip).limit(limit).order_by(desc(Deal.updated_at))
    
    result = await db.execute(query)
    return result.scalars().all()


async def get_public_deals(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> List[Deal]:
    """
    Get list of publicly accessible deals
    
    Args:
        db: Database session
        skip: Records to skip (pagination)
        limit: Maximum records to return
        filter_criteria: Optional filtering criteria
        
    Returns:
        List of public Deal objects
    """
    query = select(Deal).where(
        and_(
            Deal.is_public == True,
            Deal.is_deleted == False
        )
    )
    
    # Apply filters if provided
    if filter_criteria:
        if "name" in filter_criteria:
            query = query.where(Deal.name.ilike(f"%{filter_criteria['name']}%"))
        
        if "deal_type" in filter_criteria:
            query = query.where(Deal.deal_type == filter_criteria["deal_type"])
    
    # Apply pagination
    query = query.offset(skip).limit(limit).order_by(desc(Deal.updated_at))
    
    result = await db.execute(query)
    return result.scalars().all()


async def get_user_deals(
    db: AsyncSession,
    user_id: UUID,
    skip: int = 0,
    limit: int = 100,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> List[Deal]:
    """
    Get list of deals owned by a specific user
    
    Args:
        db: Database session
        user_id: ID of the user
        skip: Records to skip (pagination)
        limit: Maximum records to return
        filter_criteria: Optional filtering criteria
        
    Returns:
        List of Deal objects owned by the user
    """
    query = select(Deal).where(
        and_(
            Deal.owner_id == user_id,
            Deal.is_deleted == False
        )
    )
    
    # Apply filters if provided
    if filter_criteria:
        if "name" in filter_criteria:
            query = query.where(Deal.name.ilike(f"%{filter_criteria['name']}%"))
        
        if "deal_type" in filter_criteria:
            query = query.where(Deal.deal_type == filter_criteria["deal_type"])
        
        if "is_public" in filter_criteria:
            query = query.where(Deal.is_public == filter_criteria["is_public"])
    
    # Apply pagination
    query = query.offset(skip).limit(limit).order_by(desc(Deal.updated_at))
    
    result = await db.execute(query)
    return result.scalars().all()


async def get_shared_deals(
    db: AsyncSession,
    user_id: UUID,
    skip: int = 0,
    limit: int = 100,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> List[Deal]:
    """
    Get list of deals shared with a specific user
    
    Args:
        db: Database session
        user_id: ID of the user
        skip: Records to skip (pagination)
        limit: Maximum records to return
        filter_criteria: Optional filtering criteria
        
    Returns:
        List of Deal objects shared with the user
    """
    query = select(Deal).join(
        DealAccess, Deal.id == DealAccess.deal_id
    ).where(
        and_(
            DealAccess.user_id == user_id,
            Deal.is_deleted == False
        )
    )
    
    # Apply filters if provided
    if filter_criteria:
        if "name" in filter_criteria:
            query = query.where(Deal.name.ilike(f"%{filter_criteria['name']}%"))
        
        if "deal_type" in filter_criteria:
            query = query.where(Deal.deal_type == filter_criteria["deal_type"])
        
        if "permission_level" in filter_criteria:
            query = query.where(DealAccess.permission_level == filter_criteria["permission_level"])
    
    # Apply pagination
    query = query.offset(skip).limit(limit).order_by(desc(Deal.updated_at))
    
    result = await db.execute(query)
    return result.scalars().all()


async def get_deal_by_id(
    db: AsyncSession,
    deal_id: UUID
) -> Optional[Deal]:
    """
    Get a specific deal by ID
    
    Args:
        db: Database session
        deal_id: ID of the deal to retrieve
        
    Returns:
        Deal object if found, None otherwise
    """
    query = select(Deal).where(
        and_(
            Deal.id == deal_id,
            Deal.is_deleted == False
        )
    )
    
    result = await db.execute(query)
    return result.scalars().first()


async def create_deal(
    db: AsyncSession,
    deal: schemas.DealCreate
) -> Deal:
    """
    Create a new deal
    
    Args:
        db: Database session
        deal: Deal creation schema with all required fields
        
    Returns:
        Created Deal object
    """
    db_deal = Deal(
        name=deal.name,
        description=deal.description,
        deal_type=deal.deal_type,
        structure=deal.structure,
        is_public=deal.is_public,
        owner_id=deal.owner_id
    )
    
    db.add(db_deal)
    await db.commit()
    await db.refresh(db_deal)
    
    # Create initial version
    db_version = DealVersion(
        deal_id=db_deal.id,
        version_number=1,
        structure=deal.structure,
        created_by_id=deal.owner_id,
        change_description="Initial version"
    )
    
    db.add(db_version)
    await db.commit()
    
    return db_deal


async def update_deal(
    db: AsyncSession,
    deal_id: UUID,
    deal_update: schemas.DealUpdate
) -> Optional[Deal]:
    """
    Update an existing deal
    
    Args:
        db: Database session
        deal_id: ID of the deal to update
        deal_update: Update schema with fields to modify
        
    Returns:
        Updated Deal object if found, None otherwise
    """
    # Get the existing deal
    query = select(Deal).where(
        and_(
            Deal.id == deal_id,
            Deal.is_deleted == False
        )
    )
    
    result = await db.execute(query)
    db_deal = result.scalars().first()
    
    if not db_deal:
        return None
        
    # Get current version before update
    current_version = db_deal.version
    
    # Update the deal with provided fields
    update_data = deal_update.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(db_deal, field, value)
    
    # Increment version
    db_deal.version += 1
    
    await db.commit()
    await db.refresh(db_deal)
    
    # Create new version entry if structure changed
    if "structure" in update_data:
        db_version = DealVersion(
            deal_id=db_deal.id,
            version_number=db_deal.version,
            structure=db_deal.structure,
            created_by_id=db_deal.owner_id,  # Consider passing user_id separately
            change_description=f"Update from version {current_version}"
        )
        
        db.add(db_version)
        await db.commit()
    
    return db_deal


async def delete_deal(
    db: AsyncSession,
    deal_id: UUID
) -> bool:
    """
    Soft delete a deal (mark as deleted)
    
    Args:
        db: Database session
        deal_id: ID of the deal to delete
        
    Returns:
        True if successfully deleted, False otherwise
    """
    # Get the existing deal
    query = select(Deal).where(
        and_(
            Deal.id == deal_id,
            Deal.is_deleted == False
        )
    )
    
    result = await db.execute(query)
    db_deal = result.scalars().first()
    
    if not db_deal:
        return False
        
    # Mark as deleted
    db_deal.is_deleted = True
    
    await db.commit()
    return True


async def hard_delete_deal(
    db: AsyncSession,
    deal_id: UUID
) -> bool:
    """
    Permanently delete a deal and all related data
    
    Args:
        db: Database session
        deal_id: ID of the deal to delete
        
    Returns:
        True if successfully deleted, False otherwise
    """
    # Get the existing deal
    query = select(Deal).where(Deal.id == deal_id)
    
    result = await db.execute(query)
    db_deal = result.scalars().first()
    
    if not db_deal:
        return False
        
    # Delete the deal (cascade will delete access and versions)
    await db.delete(db_deal)
    await db.commit()
    
    return True


# Deal access management functions

async def create_deal_access(
    db: AsyncSession,
    access: schemas.DealAccessCreate
) -> DealAccess:
    """
    Create a new deal access grant
    
    Args:
        db: Database session
        access: Access creation schema
        
    Returns:
        Created DealAccess object
    """
    # Check if access already exists
    query = select(DealAccess).where(
        and_(
            DealAccess.deal_id == access.deal_id,
            DealAccess.user_id == access.user_id
        )
    )
    
    result = await db.execute(query)
    existing_access = result.scalars().first()
    
    if existing_access:
        # Update existing access
        existing_access.permission_level = access.permission_level
        await db.commit()
        await db.refresh(existing_access)
        return existing_access
    
    # Create new access
    db_access = DealAccess(
        deal_id=access.deal_id,
        user_id=access.user_id,
        permission_level=access.permission_level
    )
    
    db.add(db_access)
    await db.commit()
    await db.refresh(db_access)
    
    return db_access


async def update_deal_access(
    db: AsyncSession,
    deal_id: UUID,
    user_id: UUID,
    access_update: schemas.DealAccessUpdate
) -> Optional[DealAccess]:
    """
    Update an existing deal access grant
    
    Args:
        db: Database session
        deal_id: ID of the deal
        user_id: ID of the user with access
        access_update: Update schema with fields to modify
        
    Returns:
        Updated DealAccess object if found, None otherwise
    """
    # Get the existing access
    query = select(DealAccess).where(
        and_(
            DealAccess.deal_id == deal_id,
            DealAccess.user_id == user_id
        )
    )
    
    result = await db.execute(query)
    db_access = result.scalars().first()
    
    if not db_access:
        return None
        
    # Update the access with provided fields
    update_data = access_update.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(db_access, field, value)
    
    await db.commit()
    await db.refresh(db_access)
    
    return db_access


async def remove_deal_access(
    db: AsyncSession,
    deal_id: UUID,
    user_id: UUID
) -> bool:
    """
    Remove a user's access to a deal
    
    Args:
        db: Database session
        deal_id: ID of the deal
        user_id: ID of the user to remove access for
        
    Returns:
        True if access was removed, False otherwise
    """
    # Get the existing access
    query = select(DealAccess).where(
        and_(
            DealAccess.deal_id == deal_id,
            DealAccess.user_id == user_id
        )
    )
    
    result = await db.execute(query)
    db_access = result.scalars().first()
    
    if not db_access:
        return False
        
    # Delete the access
    await db.delete(db_access)
    await db.commit()
    
    return True


async def remove_all_deal_access(
    db: AsyncSession,
    deal_id: UUID
) -> int:
    """
    Remove all access grants for a deal
    
    Args:
        db: Database session
        deal_id: ID of the deal
        
    Returns:
        Number of access grants removed
    """
    # Get all access grants for the deal
    query = select(DealAccess).where(DealAccess.deal_id == deal_id)
    
    result = await db.execute(query)
    access_grants = result.scalars().all()
    
    count = 0
    for grant in access_grants:
        await db.delete(grant)
        count += 1
    
    if count > 0:
        await db.commit()
    
    return count


async def check_deal_access(
    db: AsyncSession,
    deal_id: UUID,
    user_id: UUID,
    required_level: Optional[str] = None
) -> bool:
    """
    Check if a user has access to a deal
    
    Args:
        db: Database session
        deal_id: ID of the deal
        user_id: ID of the user
        required_level: Optional minimum permission level required
        
    Returns:
        True if user has access, False otherwise
    """
    # Get the deal
    deal_query = select(Deal).where(Deal.id == deal_id)
    deal_result = await db.execute(deal_query)
    deal = deal_result.scalars().first()
    
    if not deal:
        return False
    
    # If user is the owner, they have full access
    if deal.owner_id == user_id:
        return True
    
    # If deal is public, everyone has read access
    if deal.is_public and (not required_level or required_level == "read"):
        return True
    
    # Check for explicit access grant
    query = select(DealAccess).where(
        and_(
            DealAccess.deal_id == deal_id,
            DealAccess.user_id == user_id
        )
    )
    
    result = await db.execute(query)
    access = result.scalars().first()
    
    if not access:
        return False
    
    # If specific level is required, check it
    if required_level:
        permission_levels = {
            "read": 1,
            "edit": 2,
            "admin": 3
        }
        
        user_level = permission_levels.get(access.permission_level, 0)
        required = permission_levels.get(required_level, 0)
        
        return user_level >= required
    
    return True


async def get_deal_access_list(
    db: AsyncSession,
    deal_id: UUID
) -> List[DealAccess]:
    """
    Get list of users with access to a deal
    
    Args:
        db: Database session
        deal_id: ID of the deal
        
    Returns:
        List of DealAccess objects
    """
    query = select(DealAccess).where(DealAccess.deal_id == deal_id)
    
    result = await db.execute(query)
    return result.scalars().all()


# Deal version management functions

async def get_deal_versions(
    db: AsyncSession,
    deal_id: UUID,
    skip: int = 0,
    limit: int = 100
) -> List[DealVersion]:
    """
    Get version history for a deal
    
    Args:
        db: Database session
        deal_id: ID of the deal
        skip: Records to skip (pagination)
        limit: Maximum records to return
        
    Returns:
        List of DealVersion objects
    """
    query = select(DealVersion).where(
        DealVersion.deal_id == deal_id
    ).order_by(desc(DealVersion.version_number))
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


async def get_deal_version(
    db: AsyncSession,
    deal_id: UUID,
    version_number: int
) -> Optional[DealVersion]:
    """
    Get a specific version of a deal
    
    Args:
        db: Database session
        deal_id: ID of the deal
        version_number: Version number to retrieve
        
    Returns:
        DealVersion object if found, None otherwise
    """
    query = select(DealVersion).where(
        and_(
            DealVersion.deal_id == deal_id,
            DealVersion.version_number == version_number
        )
    )
    
    result = await db.execute(query)
    return result.scalars().first()


async def create_deal_version(
    db: AsyncSession,
    version: schemas.DealVersionCreate
) -> DealVersion:
    """
    Create a new version of a deal
    
    Args:
        db: Database session
        version: Version creation schema
        
    Returns:
        Created DealVersion object
    """
    db_version = DealVersion(
        deal_id=version.deal_id,
        version_number=version.version_number,
        structure=version.structure,
        created_by_id=version.created_by_id,
        change_description=version.change_description
    )
    
    db.add(db_version)
    await db.commit()
    await db.refresh(db_version)
    
    return db_version

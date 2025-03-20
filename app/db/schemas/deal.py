"""
Pydantic schemas for the AbsBox Deal Library functionality.

These schemas provide:
1. Type validation for API requests and responses
2. Database interaction models for CRUD operations
3. Documentation for the API through OpenAPI
"""
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, validator
import json


# Base Deal schema (common attributes)
class DealBase(BaseModel):
    """Base schema for Deal with common attributes"""
    name: str = Field(..., description="Name of the deal")
    description: Optional[str] = Field(None, description="Optional description of the deal")
    deal_type: str = Field(..., description="Type of deal (e.g., 'abs', 'clo', 'rmbs', 'custom')")
    is_public: bool = Field(False, description="Whether the deal is publicly accessible")


# Schema for creating a new deal
class DealCreate(DealBase):
    """Schema for creating a new deal"""
    structure: Dict[str, Any] = Field(..., description="Deal structure in AbsBox-compatible format")
    owner_id: UUID = Field(..., description="ID of the user who owns this deal")


# Schema for updating an existing deal
class DealUpdate(BaseModel):
    """Schema for updating an existing deal"""
    name: Optional[str] = Field(None, description="Updated name of the deal")
    description: Optional[str] = Field(None, description="Updated description")
    deal_type: Optional[str] = Field(None, description="Updated deal type")
    structure: Optional[Dict[str, Any]] = Field(None, description="Updated deal structure")
    is_public: Optional[bool] = Field(None, description="Updated public status")


# Schema for deal in database
class DealInDB(DealBase):
    """Schema for a deal as stored in the database"""
    id: UUID
    structure: Dict[str, Any]
    owner_id: UUID
    created_at: datetime
    updated_at: datetime
    version: int = 1
    is_deleted: bool = False

    class Config:
        orm_mode = True


# Deal schema for API response
class Deal(DealInDB):
    """Complete Deal schema for API responses"""
    pass


# Schema for simplified deal listing (without full structure)
class DealSummary(BaseModel):
    """Simplified Deal representation for listings"""
    id: UUID
    name: str
    description: Optional[str]
    deal_type: str
    is_public: bool
    owner_id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


# Base schema for Deal Access
class DealAccessBase(BaseModel):
    """Base schema for Deal Access permissions"""
    deal_id: UUID = Field(..., description="ID of the deal being shared")
    user_id: UUID = Field(..., description="ID of the user receiving access")
    permission_level: str = Field("read", description="Permission level: read, edit, or admin")
    
    @validator('permission_level')
    def validate_permission_level(cls, v):
        allowed = ["read", "edit", "admin"]
        if v not in allowed:
            raise ValueError(f"Permission level must be one of: {', '.join(allowed)}")
        return v


# Schema for creating new access grants
class DealAccessCreate(DealAccessBase):
    """Schema for creating a new access grant"""
    pass


# Schema for updating access grants
class DealAccessUpdate(BaseModel):
    """Schema for updating an access grant"""
    permission_level: str = Field(..., description="Updated permission level")
    
    @validator('permission_level')
    def validate_permission_level(cls, v):
        allowed = ["read", "edit", "admin"]
        if v not in allowed:
            raise ValueError(f"Permission level must be one of: {', '.join(allowed)}")
        return v


# Schema for access grant in database
class DealAccessInDB(DealAccessBase):
    """Schema for an access grant as stored in the database"""
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


# Access grant schema for API response
class DealAccess(DealAccessInDB):
    """Complete Access Grant schema for API responses"""
    pass


# Schema for deal version history
class DealVersionBase(BaseModel):
    """Base schema for Deal Version history"""
    deal_id: UUID = Field(..., description="ID of the main deal")
    version_number: int = Field(..., description="Version number of this snapshot")
    change_description: Optional[str] = Field(None, description="Description of changes in this version")


# Schema for creating a new version
class DealVersionCreate(DealVersionBase):
    """Schema for creating a new deal version"""
    structure: Dict[str, Any] = Field(..., description="Deal structure at this version")
    created_by_id: Optional[UUID] = Field(None, description="User who created this version")


# Schema for version in database
class DealVersionInDB(DealVersionBase):
    """Schema for a deal version as stored in the database"""
    id: UUID
    structure: Dict[str, Any]
    created_by_id: Optional[UUID]
    created_at: datetime

    class Config:
        orm_mode = True


# Deal version schema for API response
class DealVersion(DealVersionInDB):
    """Complete Deal Version schema for API responses"""
    pass


# Schema for listing available deal versions
class DealVersionSummary(BaseModel):
    """Simplified Deal Version for listings"""
    id: UUID
    deal_id: UUID
    version_number: int
    change_description: Optional[str]
    created_by_id: Optional[UUID]
    created_at: datetime
    
    class Config:
        orm_mode = True

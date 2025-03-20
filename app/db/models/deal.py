"""
Database models for the AbsBox Deal Library implementation.

These models support:
1. Public deals accessible to all users
2. Private deals owned by specific users
3. Shared private deals with permissions
"""
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Text, Enum, JSON, Table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base

class Deal(Base):
    """
    Deal model for storing structured finance deals in the AbsBox Deal Library
    """
    __tablename__ = "deals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    deal_type = Column(String, nullable=False, index=True)
    
    # JSON structure for the deal configuration (compatible with AbsBox format)
    structure = Column(JSON, nullable=False)
    
    # Public deals are accessible to all users
    is_public = Column(Boolean, default=False, index=True)
    
    # Deal ownership
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    owner = relationship("User", back_populates="owned_deals")
    
    # Timestamps for auditing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Access permissions for shared deals
    access_grants = relationship("DealAccess", back_populates="deal", cascade="all, delete-orphan")
    
    # Versioning and history
    version = Column(Integer, default=1)
    is_deleted = Column(Boolean, default=False, index=True)

    def __repr__(self):
        return f"<Deal {self.name} (ID: {self.id})>"


class DealAccess(Base):
    """
    Deal access permissions for shared deals
    
    This allows private deals to be shared with specific users with different
    permission levels (read, edit, admin)
    """
    __tablename__ = "deal_access"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # The deal being shared
    deal_id = Column(UUID(as_uuid=True), ForeignKey("deals.id", ondelete="CASCADE"), nullable=False)
    deal = relationship("Deal", back_populates="access_grants")
    
    # The user receiving access
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    user = relationship("User", back_populates="deal_access")
    
    # Permission level (read, edit, admin)
    permission_level = Column(Enum("read", "edit", "admin", name="permission_level_enum"), 
                            default="read", nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    class Config:
        # Create a unique constraint to prevent duplicate access grants
        __table_args__ = (
            sqlalchemy.UniqueConstraint('deal_id', 'user_id', name='uq_deal_user'),
        )

    def __repr__(self):
        return f"<DealAccess Deal: {self.deal_id}, User: {self.user_id}, Level: {self.permission_level}>"


class DealVersion(Base):
    """
    Deal version history for tracking changes
    
    This stores previous versions of deals to enable history tracking,
    auditing, and rollback capability.
    """
    __tablename__ = "deal_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Reference to the main deal
    deal_id = Column(UUID(as_uuid=True), ForeignKey("deals.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Version metadata
    version_number = Column(Integer, nullable=False)
    structure = Column(JSON, nullable=False)  # The deal structure at this version
    
    # User who created this version
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Change metadata
    change_description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<DealVersion {self.deal_id} v{self.version_number}>"

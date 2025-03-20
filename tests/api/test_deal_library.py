"""
Tests for the Deal Library API endpoints.

These tests ensure that the deal library functionality works correctly,
including permissions, sharing, and integration with HAStructure.
"""
import pytest
import uuid
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
import json

from app.main import app
from app.db.models.deal import Deal, DealAccess
from app.db.schemas.deal import DealCreate, DealUpdate

# Test data
test_deal_data = {
    "name": "Test ABS Deal",
    "description": "Test deal for unit tests",
    "deal_type": "ABS",
    "structure": {
        "pool": [
            {
                "balance": 1000000,
                "rate": 0.05,
                "term": 60,
                "type": "fixed"
            }
        ],
        "liabilities": [
            {
                "balance": 900000,
                "rate": 0.03,
                "name": "Class A"
            },
            {
                "balance": 100000,
                "rate": 0.06,
                "name": "Class B"
            }
        ],
        "waterfall": {
            "normal": [
                {"type": "interest", "source": "pool", "target": "Class A"},
                {"type": "interest", "source": "pool", "target": "Class B"},
                {"type": "principal", "source": "pool", "target": "Class A"},
                {"type": "principal", "source": "pool", "target": "Class B"}
            ]
        }
    },
    "is_public": False
}


# Fixtures
@pytest.fixture
def client():
    """Test client for API requests"""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def db_session():
    """Provides database session for tests"""
    from app.db.session import get_db, engine_test
    from app.db.base import Base

    # Create all tables in test database
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Provide session
    db = get_db()
    try:
        db_session = await anext(db)
        yield db_session
    finally:
        await db_session.close()
    
    # Drop all tables after tests
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def test_user():
    """Create test user"""
    from app.db.models.user import User
    
    return {
        "id": uuid.uuid4(),
        "email": "test_user@example.com",
        "full_name": "Test User"
    }


@pytest.fixture
async def test_user2():
    """Create second test user for sharing tests"""
    from app.db.models.user import User
    
    return {
        "id": uuid.uuid4(),
        "email": "test_user2@example.com",
        "full_name": "Test User 2"
    }


@pytest.fixture
async def auth_headers(test_user):
    """Create auth headers for test user"""
    from app.core.security import create_access_token
    
    access_token = create_access_token(data={"sub": str(test_user["id"])})
    
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
async def auth_headers2(test_user2):
    """Create auth headers for second test user"""
    from app.core.security import create_access_token
    
    access_token = create_access_token(data={"sub": str(test_user2["id"])})
    
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
async def test_deal(db_session, test_user):
    """Create a test deal owned by test_user"""
    deal = Deal(
        id=uuid.uuid4(),
        name=test_deal_data["name"],
        description=test_deal_data["description"],
        deal_type=test_deal_data["deal_type"],
        structure=test_deal_data["structure"],
        is_public=test_deal_data["is_public"],
        owner_id=test_user["id"]
    )
    
    db_session.add(deal)
    await db_session.commit()
    await db_session.refresh(deal)
    
    yield deal
    
    # Cleanup
    await db_session.delete(deal)
    await db_session.commit()


@pytest.fixture
async def test_public_deal(db_session, test_user):
    """Create a public test deal"""
    deal = Deal(
        id=uuid.uuid4(),
        name="Public Test Deal",
        description="Public test deal for unit tests",
        deal_type="ABS",
        structure=test_deal_data["structure"],
        is_public=True,
        owner_id=test_user["id"]
    )
    
    db_session.add(deal)
    await db_session.commit()
    await db_session.refresh(deal)
    
    yield deal
    
    # Cleanup
    await db_session.delete(deal)
    await db_session.commit()


@pytest.fixture
async def test_shared_deal(db_session, test_user, test_user2):
    """Create a deal shared between test_user and test_user2"""
    # Create the deal
    deal = Deal(
        id=uuid.uuid4(),
        name="Shared Test Deal",
        description="Shared test deal for unit tests",
        deal_type="ABS",
        structure=test_deal_data["structure"],
        is_public=False,
        owner_id=test_user["id"]
    )
    
    db_session.add(deal)
    await db_session.commit()
    
    # Create access permission
    access = DealAccess(
        deal_id=deal.id,
        user_id=test_user2["id"],
        permission_level="read"
    )
    
    db_session.add(access)
    await db_session.commit()
    await db_session.refresh(deal)
    
    yield deal
    
    # Cleanup
    await db_session.delete(access)
    await db_session.delete(deal)
    await db_session.commit()


# Tests
@pytest.mark.asyncio
async def test_create_deal(client, auth_headers, test_user):
    """Test creating a new deal"""
    response = client.post(
        "/api/v1/deals",
        json=test_deal_data,
        headers=auth_headers,
        params={"is_public": False}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == test_deal_data["name"]
    assert data["owner_id"] == str(test_user["id"])
    assert "id" in data
    
    # Clean up
    client.delete(f"/api/v1/deals/{data['id']}", headers=auth_headers)


@pytest.mark.asyncio
async def test_get_deal(client, auth_headers, test_deal):
    """Test getting a specific deal"""
    response = client.get(
        f"/api/v1/deals/{test_deal.id}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(test_deal.id)
    assert data["name"] == test_deal.name
    assert data["owner_id"] == str(test_deal.owner_id)


@pytest.mark.asyncio
async def test_get_public_deal_unauthenticated(client, test_public_deal):
    """Test getting a public deal without authentication"""
    response = client.get(f"/api/v1/deals/{test_public_deal.id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(test_public_deal.id)
    assert data["is_public"] == True


@pytest.mark.asyncio
async def test_get_private_deal_unauthenticated(client, test_deal):
    """Test getting a private deal without authentication (should fail)"""
    response = client.get(f"/api/v1/deals/{test_deal.id}")
    
    assert response.status_code == 403  # Forbidden


@pytest.mark.asyncio
async def test_list_public_deals(client):
    """Test listing public deals"""
    response = client.get("/api/v1/deals/public")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    
    # Check that all deals are public
    for deal in data:
        assert deal["is_public"] == True


@pytest.mark.asyncio
async def test_list_my_deals(client, auth_headers, test_deal, test_user):
    """Test listing user's own deals"""
    response = client.get(
        "/api/v1/deals/my-deals",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    
    # Check that all deals belong to the user
    for deal in data:
        assert deal["owner_id"] == str(test_user["id"])
    
    # Check that test_deal is in the list
    deal_ids = [deal["id"] for deal in data]
    assert str(test_deal.id) in deal_ids


@pytest.mark.asyncio
async def test_list_shared_deals(client, auth_headers2, test_shared_deal, test_user2):
    """Test listing deals shared with the user"""
    response = client.get(
        "/api/v1/deals/shared-with-me",
        headers=auth_headers2
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    
    # Check that the shared deal is in the list
    deal_ids = [deal["id"] for deal in data]
    assert str(test_shared_deal.id) in deal_ids


@pytest.mark.asyncio
async def test_update_deal(client, auth_headers, test_deal):
    """Test updating a deal"""
    update_data = {
        "name": "Updated Deal Name",
        "description": "Updated description"
    }
    
    response = client.put(
        f"/api/v1/deals/{test_deal.id}",
        json=update_data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["description"] == update_data["description"]
    assert data["id"] == str(test_deal.id)


@pytest.mark.asyncio
async def test_update_deal_unauthorized(client, auth_headers2, test_deal):
    """Test unauthorized update (should fail)"""
    update_data = {
        "name": "Unauthorized Update"
    }
    
    response = client.put(
        f"/api/v1/deals/{test_deal.id}",
        json=update_data,
        headers=auth_headers2
    )
    
    assert response.status_code == 403  # Forbidden


@pytest.mark.asyncio
async def test_delete_deal(client, auth_headers, db_session, test_user):
    """Test deleting a deal"""
    # Create a deal to delete
    deal = Deal(
        id=uuid.uuid4(),
        name="Deal to Delete",
        description="This deal will be deleted",
        deal_type="ABS",
        structure=test_deal_data["structure"],
        is_public=False,
        owner_id=test_user["id"]
    )
    
    db_session.add(deal)
    await db_session.commit()
    await db_session.refresh(deal)
    
    # Delete the deal
    response = client.delete(
        f"/api/v1/deals/{deal.id}",
        headers=auth_headers
    )
    
    assert response.status_code == 204
    
    # Verify it's deleted
    get_response = client.get(
        f"/api/v1/deals/{deal.id}",
        headers=auth_headers
    )
    
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_share_deal(client, auth_headers, test_deal, test_user2):
    """Test sharing a deal with another user"""
    share_data = {
        "shared_with_id": str(test_user2["id"]),
        "permission_level": "read"
    }
    
    response = client.post(
        f"/api/v1/deals/{test_deal.id}/share",
        json=share_data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    assert response.json() == True
    
    # Verify the user can access the deal
    client2 = TestClient(app)
    from app.core.security import create_access_token
    access_token2 = create_access_token(data={"sub": str(test_user2["id"])})
    headers2 = {"Authorization": f"Bearer {access_token2}"}
    
    get_response = client2.get(
        f"/api/v1/deals/{test_deal.id}",
        headers=headers2
    )
    
    assert get_response.status_code == 200


@pytest.mark.asyncio
async def test_revoke_access(client, auth_headers, test_shared_deal, test_user2):
    """Test revoking a user's access to a deal"""
    response = client.delete(
        f"/api/v1/deals/{test_shared_deal.id}/share/{test_user2['id']}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    assert response.json() == True
    
    # Verify the user can no longer access the deal
    client2 = TestClient(app)
    from app.core.security import create_access_token
    access_token2 = create_access_token(data={"sub": str(test_user2["id"])})
    headers2 = {"Authorization": f"Bearer {access_token2}"}
    
    get_response = client2.get(
        f"/api/v1/deals/{test_shared_deal.id}",
        headers=headers2
    )
    
    assert get_response.status_code == 403  # Forbidden


@pytest.mark.asyncio
async def test_get_deal_access_list(client, auth_headers, test_shared_deal):
    """Test getting the list of users with access to a deal"""
    response = client.get(
        f"/api/v1/deals/{test_shared_deal.id}/access",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # At least one access entry should exist (the one we created in the fixture)
    access_found = False
    for access in data:
        if "user_id" in access and "permission_level" in access:
            access_found = True
            break
    
    assert access_found


@pytest.mark.asyncio
async def test_export_deal(client, auth_headers, test_deal):
    """Test exporting a deal to AbsBox format"""
    response = client.post(
        f"/api/v1/deals/{test_deal.id}/export",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify the expected structure of the exported deal
    assert "pool" in data
    assert "liabilities" in data
    assert "waterfall" in data


@pytest.mark.asyncio
async def test_import_deal(client, auth_headers, test_user):
    """Test importing a deal from AbsBox format"""
    import_data = {
        "deal_data": test_deal_data["structure"],
        "name": "Imported Deal",
        "description": "Deal imported from AbsBox format"
    }
    
    response = client.post(
        "/api/v1/deals/import",
        json=import_data,
        headers=auth_headers,
        params={"is_public": True}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == import_data["name"]
    assert data["owner_id"] == str(test_user["id"])
    assert data["is_public"] == True
    
    # Cleanup
    client.delete(f"/api/v1/deals/{data['id']}", headers=auth_headers)


@pytest.mark.asyncio
async def test_clone_deal(client, auth_headers, test_deal, test_user):
    """Test cloning a deal"""
    response = client.post(
        f"/api/v1/deals/{test_deal.id}/clone",
        headers=auth_headers,
        params={
            "new_name": "Cloned Deal",
            "new_description": "This is a cloned deal",
            "is_public": True
        }
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Cloned Deal"
    assert data["description"] == "This is a cloned deal"
    assert data["owner_id"] == str(test_user["id"])
    assert data["is_public"] == True
    
    # Verify the structure was copied
    assert data["structure"] == test_deal_data["structure"]
    
    # Cleanup
    client.delete(f"/api/v1/deals/{data['id']}", headers=auth_headers)

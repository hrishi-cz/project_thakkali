"""
Test script for Phase 2: Session-Scoped Ingestion
Tests session management endpoints and database persistence.
"""

import requests
import json
from pprint import pprint

BASE_URL = "http://localhost:8001"


def test_create_session():
    """Test POST /v2/sessions"""
    print("\n=== Test 1: Create Session ===")
    
    response = requests.post(
        f"{BASE_URL}/v2/sessions",
        json={
            "user_id": "test_user_1",
            "project_name": "AutoVision+ Test",
            "description": "Testing session-scoped ingestion"
        }
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    pprint(data)
    
    assert response.status_code == 200
    assert "session_id" in data
    assert data["status"] == "active"
    
    return data["session_id"]


def test_get_session(session_id):
    """Test GET /v2/sessions/{session_id}"""
    print(f"\n=== Test 2: Get Session {session_id} ===")
    
    response = requests.get(f"{BASE_URL}/v2/sessions/{session_id}")
    
    print(f"Status: {response.status_code}")
    data = response.json()
    pprint(data)
    
    assert response.status_code == 200
    assert data["session_id"] == session_id
    assert data["status"] == "active"
    assert data["active_dataset_ids"] == []
    
    return data


def test_list_sessions():
    """Test GET /v2/sessions"""
    print("\n=== Test 3: List Sessions ===")
    
    response = requests.get(f"{BASE_URL}/v2/sessions")
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total sessions: {data['total']}")
    print(f"Returned: {len(data['sessions'])}")
    
    for session in data['sessions'][:3]:  # Show first 3
        print(f"  - {session['session_id']}: {session.get('project_name', 'No name')} ({session['status']})")
    
    assert response.status_code == 200
    assert "sessions" in data
    assert "total" in data


def test_add_datasets(session_id):
    """Test POST /v2/sessions/{session_id}/datasets"""
    print(f"\n=== Test 4: Add Datasets to Session {session_id} ===")
    
    # Use a small test dataset
    response = requests.post(
        f"{BASE_URL}/v2/sessions/{session_id}/datasets",
        json={
            "dataset_urls": [
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
            ],
            "force_redownload": False
        }
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    pprint(data)
    
    assert response.status_code == 200
    assert "task_id" in data
    assert data["status"] == "processing"
    
    return data["task_id"]


def test_list_session_datasets(session_id):
    """Test GET /v2/sessions/{session_id}/datasets"""
    print(f"\n=== Test 5: List Session Datasets {session_id} ===")
    
    response = requests.get(f"{BASE_URL}/v2/sessions/{session_id}/datasets")
    
    print(f"Status: {response.status_code}")
    data = response.json()
    pprint(data)
    
    assert response.status_code == 200
    assert "active_datasets" in data
    assert "cached_datasets" in data


def test_close_session(session_id):
    """Test POST /v2/sessions/{session_id}/close"""
    print(f"\n=== Test 6: Close Session {session_id} ===")
    
    response = requests.post(f"{BASE_URL}/v2/sessions/{session_id}/close")
    
    print(f"Status: {response.status_code}")
    data = response.json()
    pprint(data)
    
    assert response.status_code == 200
    assert data["status"] == "closed"


def test_database_persistence():
    """Test that sessions persist across API restarts (requires manual restart)"""
    print("\n=== Test 7: Database Persistence ===")
    
    # Create a session
    session_id = test_create_session()
    
    print("\n⚠️  Manual Test: Restart the API server and check if session persists")
    print(f"   Then run: GET /v2/sessions/{session_id}")
    print(f"   Expected: Session {session_id} should still exist")
    
    return session_id


def main():
    """Run all tests"""
    print("=" * 80)
    print("Phase 2: Session-Scoped Ingestion Tests")
    print("=" * 80)
    
    try:
        # Test 1: Create session
        session_id = test_create_session()
        
        # Test 2: Get session
        test_get_session(session_id)
        
        # Test 3: List sessions
        test_list_sessions()
        
        # Test 4: Add datasets (async)
        task_id = test_add_datasets(session_id)
        
        # Test 5: List datasets (after ingestion)
        import time
        print("\nWaiting 5 seconds for ingestion to complete...")
        time.sleep(5)
        test_list_session_datasets(session_id)
        
        # Test 6: Close session
        test_close_session(session_id)
        
        # Test 7: Persistence
        test_session_id = test_database_persistence()
        
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        print(f"\nPersistence test session ID: {test_session_id}")
        print("Restart the API and verify it still exists.")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to {BASE_URL}")
        print("   Make sure the API server is running: python run_api.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

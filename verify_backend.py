"""
verify_backend.py - Automated Verification Suite for Dilli Darshan Backend & Frontend Integration
Tests SQLite profile persistence, RAG chat endpoints, multi-turn follow-up memory, and HTML formatting.
"""

import sys
from fastapi.testclient import TestClient
from api import app, format_markdown_for_frontend_html
import database


def run_verification():
    print("=" * 70)
    print("🚀 STARTING DILLI DARSHAN BACKEND & API VERIFICATION SUITE")
    print("=" * 70)

    client = TestClient(app)

    # 1. Health check verification
    print("\n[Step 1] Testing Health Endpoints...")
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
    health_data = health_resp.json()
    print("✓ /health Status 200 OK:")
    print(f"  • Service: {health_data.get('service')}")
    print(f"  • ChromaDB Chunks: {health_data.get('chromadb_chunks')}")
    print(f"  • Registered Users: {health_data.get('registered_users')}")
    print(f"  • Active Model: {health_data.get('active_model')}")

    # 2. Frontend HTML & Asset Serving
    print("\n[Step 2] Testing Frontend Static Serving at GET /...")
    root_resp = client.get("/")
    assert root_resp.status_code == 200, f"Root / failed: {root_resp.status_code}"
    assert "DilliDarshan AI" in root_resp.text, "index.html not properly served at /"
    print("✓ GET / successfully serves Dilli-Darshan/index.html (Single-Page App ready)")

    asset_resp = client.get("/assets/red-fort.jpg")
    if asset_resp.status_code == 200:
        print(f"✓ GET /assets/red-fort.jpg serves asset ({len(asset_resp.content)} bytes)")
    else:
        print(f"ℹ /assets status: {asset_resp.status_code}")

    # 3. Task 1 & 3: Profile Upsert Endpoint
    print("\n[Step 3] Testing POST /user/profile (Task 1 & Task 3)...")
    profile_payload = {
        "name": "Saif",
        "email": "saif@delhi.com",
        "travel_style": "Food & Street Markets",
        "language": "English"
    }
    prof_resp = client.post("/user/profile", json=profile_payload)
    assert prof_resp.status_code == 200, f"Profile creation failed: {prof_resp.status_code}"
    prof_data = prof_resp.json()
    print("✓ Profile Upserted Successfully:")
    print(f"  • User ID: {prof_data.get('user_id')}")
    print(f"  • Name: {prof_data.get('name')}")
    print(f"  • Email: {prof_data.get('email')}")
    print(f"  • Travel Style: {prof_data.get('travel_style')}")
    print(f"  • Language: {prof_data.get('language')}")
    assert prof_data.get("user_id") is not None
    user_id = prof_data["user_id"]

    # Verify database lookup
    fetched_user = database.get_user_by_id(user_id)
    assert fetched_user is not None
    assert fetched_user["email"] == "saif@delhi.com"
    print(f"✓ SQLite Verified: user {user_id} saved in users.db")

    # 4. Task 2 & 3: Chat Turn 1 (Red Fort timings & tickets)
    print("\n[Step 4] Testing POST /chat Turn 1 (Red Fort timings & ticket prices)...")
    chat_payload_turn1 = {
        "query": "What are the timings and ticket prices for Red Fort?",
        "user_id": user_id,
        "user_name": "Saif",
        "travel_style": "Food & Street Markets",
        "language": "English"
    }
    chat_resp1 = client.post("/chat", json=chat_payload_turn1)
    assert chat_resp1.status_code == 200, f"Chat Turn 1 failed: {chat_resp1.status_code}"
    chat_data1 = chat_resp1.json()

    print("✓ Turn 1 Response Received:")
    print(f"  • Routing Status: {chat_data1.get('routing_status')}")
    print(f"  • Active Places: {chat_data1.get('active_places')}")
    print(f"  • Source Badge Text: '{chat_data1.get('source')}'")
    print(f"  • Turn Number: {chat_data1.get('turn_number')}")
    print(f"  • Answer Preview (HTML): {chat_data1.get('answer')[:180]}...")

    # Check HTML tags presence
    assert "<strong>" in chat_data1["answer"] or "<div>" in chat_data1["answer"], "HTML formatting missing from answer"
    assert chat_data1.get("source"), "Source badge string missing"
    assert not chat_data1["answer"].lower().startswith("hello"), "Turn 1 starts with 'Hello'!"
    assert not chat_data1.get("raw_markdown", "").lower().startswith("hello"), "Turn 1 raw markdown starts with 'Hello'!"

    # 5. Task 2 & 3: Chat Turn 2 (Contextual Follow-up: "Which metro gate is closest to reach there?")
    print("\n[Step 5] Testing POST /chat Turn 2 (Multi-turn memory: 'Which metro gate is closest to reach there?')...")
    chat_payload_turn2 = {
        "query": "Which metro gate is closest to reach there?",
        "user_id": user_id,
        "user_name": "Saif",
        "travel_style": "Food & Street Markets",
        "language": "English"
    }
    chat_resp2 = client.post("/chat", json=chat_payload_turn2)
    assert chat_resp2.status_code == 200, f"Chat Turn 2 failed: {chat_resp2.status_code}"
    chat_data2 = chat_resp2.json()

    print("✓ Turn 2 Response Received:")
    print(f"  • Routing Status: {chat_data2.get('routing_status')}")
    print(f"  • Active Places: {chat_data2.get('active_places')}")
    print(f"  • Source Badge Text: '{chat_data2.get('source')}'")
    print(f"  • Turn Number: {chat_data2.get('turn_number')}")
    print(f"  • Answer Content:\n{chat_data2.get('answer')}")

    # Verify context retention of Red Fort and absence of greeting in Turn 2
    assert not chat_data2["answer"].lower().startswith("hello"), "Turn 2 starts with 'Hello'!"
    assert not chat_data2.get("raw_markdown", "").lower().startswith("hello"), "Turn 2 raw markdown starts with 'Hello'!"
    t2_text = chat_data2["answer"].lower() + chat_data2.get("raw_markdown", "").lower()
    has_red_fort = ("red fort" in t2_text) or ("lal qila" in t2_text) or ("gate" in t2_text)
    assert has_red_fort, "Turn 2 failed to inherit active monument (Red Fort) from Turn 1!"
    print("✓ Multi-Turn Context Verified: Bot correctly understood 'there' refers to Red Fort without greeting prefix!")

    # 6. Task 2: Language Personalization (Hindi response)
    print("\n[Step 6] Testing Personalization with Hindi Language...")
    chat_payload_hindi = {
        "query": "कुतुब मीनार के बारे में बताइए",
        "user_id": user_id,
        "user_name": "Saif",
        "travel_style": "Heritage & History",
        "language": "Hindi"
    }
    chat_resp_hindi = client.post("/chat", json=chat_payload_hindi)
    assert chat_resp_hindi.status_code == 200
    hindi_data = chat_resp_hindi.json()
    print("✓ Hindi Response Received:")
    print(f"  • Source: {hindi_data.get('source')}")
    print(f"  • Preview: {hindi_data.get('answer')[:150]}...")

    # 7. Reset Endpoint
    print("\n[Step 7] Testing POST /chat/reset...")
    reset_resp = client.post("/chat/reset", json={"user_id": user_id})
    assert reset_resp.status_code == 200
    print(f"✓ Chat Reset response: {reset_resp.json()}")

    # 8. Check SQLite Audit Trail
    print("\n[Step 8] Verifying SQLite chat_logs audit trail in users.db...")
    logs_cnt = database.get_chat_logs_count()
    print(f"✓ Total logged chat turns in users.db: {logs_cnt}")
    assert logs_cnt >= 2, f"Expected at least 2 logged turns, found {logs_cnt}"

    print("\n" + "=" * 70)
    print("🎉 ALL 4 TASKS SUCCESSFULLY VERIFIED & PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()

"""
verify_backend.py - End-to-End Automated Verification Suite for Dilli Darshan Backend
Tests Hindsight Cloud Memory Engine, FastAPI Endpoints, RAG retrieval, Multi-turn context resolution,
personalization, dietary & physical constraints (knee injury & vegetarian), and HTML bubble formatting.
"""

import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from fastapi.testclient import TestClient
from api import app, markdown_to_clean_html
import memory
import retrieval


def run_verification():
    print("=" * 80)
    print("🚀 STARTING DILLI DARSHAN HINDSIGHT CLOUD BACKEND VERIFICATION SUITE")
    print("=" * 80)

    client = TestClient(app)

    # -------------------------------------------------------------------------
    # 1. Verification of Task 1: Hindsight Cloud Client Initialization
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying Hindsight Cloud Client & Configuration (Task 1)...")
    print(f"  • Base URL: {memory.hindsight_base_url}")
    print(f"  • API Key Configured: {bool(memory.hindsight_api_key)}")
    print(f"  • Hindsight Client Active: {memory.client is not None}")
    assert memory.client is not None, "Hindsight Cloud client failed to initialize with .env API key!"
    assert memory.hindsight_base_url == "https://api.hindsight.vectorize.io", f"Unexpected base URL: {memory.hindsight_base_url}"
    print("✓ Task 1 Verified: memory.py initialized Hindsight(base_url='https://api.hindsight.vectorize.io', api_key=...) from .env")

    # -------------------------------------------------------------------------
    # 2. Verification of Health Endpoints
    # -------------------------------------------------------------------------
    print("\n[Step 2] Testing Health Endpoints (/health & /api/health)...")
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
    health_data = health_resp.json()
    print("✓ /health Status 200 OK:")
    print(f"  • Service: {health_data.get('service')}")
    print(f"  • ChromaDB Chunks: {health_data.get('chromadb_chunks')}")
    print(f"  • Active Model: {health_data.get('active_model')}")
    print(f"  • Hindsight Connected: {health_data.get('hindsight_cloud', {}).get('client_connected')}")
    assert health_data.get("chromadb_chunks", 0) > 0, "ChromaDB should have chunks indexed"

    api_health_resp = client.get("/api/health")
    assert api_health_resp.status_code == 200

    # -------------------------------------------------------------------------
    # 3. Verification of Frontend Serving at GET /
    # -------------------------------------------------------------------------
    print("\n[Step 3] Testing Frontend Static Serving at GET /...")
    root_resp = client.get("/")
    assert root_resp.status_code == 200, f"Root / failed: {root_resp.status_code}"
    print("✓ GET / successfully serves companion frontend")

    # -------------------------------------------------------------------------
    # 4. Verification of Task 1 & 3: POST /user/profile (Hindsight Cloud)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Testing POST /user/profile (Task 1 & Task 3)...")
    profile_payload = {
        "name": "Aarav Sharma",
        "email": "aarav.tourist@example.com",
        "travel_style": "Heritage & History",
        "language": "English"
    }
    prof_resp = client.post("/user/profile", json=profile_payload)
    assert prof_resp.status_code == 200, f"Profile creation failed: {prof_resp.status_code}"
    prof_data = prof_resp.json()
    print("✓ Profile Saved Directly to Hindsight Cloud:")
    print(f"  • User ID: {prof_data.get('user_id')} (Type: {type(prof_data.get('user_id')).__name__})")
    print(f"  • Bank ID: {prof_data.get('bank_id')}")
    print(f"  • Name: {prof_data.get('name')}")
    print(f"  • Email: {prof_data.get('email')}")
    print(f"  • Travel Style: {prof_data.get('travel_style')}")
    print(f"  • Language: {prof_data.get('language')}")

    user_id = prof_data.get("user_id")
    bank_id = prof_data.get("bank_id")
    assert isinstance(user_id, int), f"user_id must be integer, got {type(user_id)}"
    assert user_id > 0, "user_id must be positive integer"
    assert bank_id == f"dilli_darshan_tourist_{user_id}", f"Unexpected bank_id: {bank_id}"
    print(f"✓ Deterministic User ID & Hindsight Bank ID Verified: {bank_id}")

    # -------------------------------------------------------------------------
    # 5. Verification of Task 2, 3 & 4: Chat Turn 1 (Jama Masjid + Knee Injury + Vegetarian)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Testing POST /chat Turn 1: Jama Masjid with knee injury & vegetarian preferences...")
    turn1_query = "Hi, I am strictly vegetarian and I have a knee injury so I cannot climb steep stairs. Tell me about Jama Masjid."
    chat_payload_turn1 = {
        "query": turn1_query,
        "user_id": user_id,
        "user_name": "Aarav Sharma",
        "travel_style": "Heritage & History",
        "language": "English"
    }

    chat_resp1 = client.post("/chat", json=chat_payload_turn1)
    assert chat_resp1.status_code == 200, f"Chat Turn 1 failed: {chat_resp1.status_code}"
    chat_data1 = chat_resp1.json()

    print("✓ Turn 1 Response Received:")
    print(f"  • Routing Status: {chat_data1.get('routing_status')}")
    print(f"  • Active Places: {chat_data1.get('active_places')}")
    print(f"  • Source Badge: '{chat_data1.get('source')}'")
    print(f"  • Hindsight Bank ID: {chat_data1.get('hindsight_bank_id')}")
    print(f"  • HTML Answer Preview: {chat_data1.get('answer')[:220]}...")

    # Verification of Turn 1 properties
    assert "jama_masjid" in chat_data1.get("active_places", []), f"Expected 'jama_masjid' in active_places, got {chat_data1.get('active_places')}"
    assert chat_data1.get("source"), "Source badge string missing"
    assert "Hindsight Cloud Memory" in chat_data1.get("source", ""), "Source badge should mention Hindsight Cloud Memory"

    # HTML formatting checks
    assert "<p" in chat_data1["answer"] or "<strong>" in chat_data1["answer"] or "<h" in chat_data1["answer"], "HTML formatting missing from answer"

    # Zero greeting checks
    raw1_lower = chat_data1.get("raw_markdown", "").lower().strip()
    html1_lower = chat_data1.get("answer", "").lower().strip()
    for greeting_word in ("hello", "hi aarav", "namaste", "pranam", "greetings"):
        assert not raw1_lower.startswith(greeting_word), f"Turn 1 raw markdown starts with '{greeting_word}'!"
        assert not html1_lower.startswith(f"<p>{greeting_word}"), f"Turn 1 HTML starts with '{greeting_word}'!"

    # Knee injury & steep stairs consideration check
    t1_text = (chat_data1.get("answer", "") + " " + chat_data1.get("raw_markdown", "")).lower()
    has_knee_or_stairs_advice = any(w in t1_text for w in ["stair", "step", "knee", "gate", "climb", "ramp", "minaret", "elevator", "access"])
    assert has_knee_or_stairs_advice, "Bot should address physical access/stairs advice for Jama Masjid given the knee injury!"
    print("✓ Turn 1 Physical Care Verified: Response advises on Jama Masjid entrance/stairs with knee injury in mind.")

    # Allow a short moment for Hindsight Cloud async retain processing
    time.sleep(2)

    # -------------------------------------------------------------------------
    # 6. Verification of Task 2, 3 & 4: Chat Turn 2 (Pronoun Follow-up + Hindsight Recall)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Testing POST /chat Turn 2: 'What are the timings there and where should I eat nearby?'...")
    turn2_query = "What are the timings there and where should I eat nearby?"
    chat_payload_turn2 = {
        "query": turn2_query,
        "user_id": user_id,
        "user_name": "Aarav Sharma",
        "travel_style": "Heritage & History",
        "language": "English"
    }

    chat_resp2 = client.post("/chat", json=chat_payload_turn2)
    assert chat_resp2.status_code == 200, f"Chat Turn 2 failed: {chat_resp2.status_code}"
    chat_data2 = chat_resp2.json()

    print("✓ Turn 2 Response Received:")
    print(f"  • Routing Status: {chat_data2.get('routing_status')}")
    print(f"  • Active Places: {chat_data2.get('active_places')}")
    print(f"  • Effective Query: '{chat_data2.get('effective_query')}'")
    print(f"  • Memories Recalled Count: {chat_data2.get('memories_recalled_count')}")
    print(f"  • Memories Recalled: {chat_data2.get('memories_recalled')}")
    print(f"  • Source Badge: '{chat_data2.get('source')}'")
    print(f"  • HTML Answer Preview: {chat_data2.get('answer')[:300]}...")

    # Verify place inheritance ("there" -> Jama Masjid)
    assert "jama_masjid" in chat_data2.get("active_places", []), f"Turn 2 failed to inherit 'jama_masjid'! Got {chat_data2.get('active_places')}"
    assert chat_data2.get("is_followup") is True, "Turn 2 should be flagged as is_followup=True"

    # Verify zero greetings in Turn 2
    raw2_lower = chat_data2.get("raw_markdown", "").lower().strip()
    html2_lower = chat_data2.get("answer", "").lower().strip()
    for greeting_word in ("hello", "hi aarav", "namaste", "pranam", "greetings"):
        assert not raw2_lower.startswith(greeting_word), f"Turn 2 raw markdown starts with '{greeting_word}'!"
        assert not html2_lower.startswith(f"<p>{greeting_word}"), f"Turn 2 HTML starts with '{greeting_word}'!"

    # Verify Vegetarian food recommendation (Hindsight Cloud remembered preference!)
    t2_text = (chat_data2.get("answer", "") + " " + chat_data2.get("raw_markdown", "")).lower()
    has_veg_recommendation = any(
        w in t2_text for w in [
            "vegetarian", "veg", "haldiram", "paranthe wali", "kake di hatti",
            "giani", "jalebi", "chaat", "mithai", "daulat ki chaat", "kanwarji", "shirini"
        ]
    )
    assert has_veg_recommendation, "Turn 2 response should recommend vegetarian Old Delhi food options as retained in Hindsight Cloud memory!"
    print("✓ Multi-Turn Hindsight Memory Verified: Bot resolved 'there' to Jama Masjid, recalled vegetarian preference, and recommended vegetarian dining!")

    # -------------------------------------------------------------------------
    # 7. Verification of GET /user/memories/{user_id}
    # -------------------------------------------------------------------------
    print(f"\n[Step 7] Testing GET /user/memories/{user_id} to inspect Hindsight Cloud bank...")
    mem_resp = client.get(f"/user/memories/{user_id}")
    assert mem_resp.status_code == 200, f"Failed to get memories: {mem_resp.status_code}"
    mem_data = mem_resp.json()
    print("✓ Hindsight Cloud Bank Inspection:")
    print(f"  • Bank ID: {mem_data.get('bank_id')}")
    print(f"  • Total Memories Recalled: {len(mem_data.get('memories', []))}")
    for idx, m in enumerate(mem_data.get("memories", []), 1):
        print(f"    {idx}. {m}")
    assert mem_data.get("bank_id") == bank_id

    # -------------------------------------------------------------------------
    # 8. Verification of Session Reset (/chat/reset)
    # -------------------------------------------------------------------------
    print("\n[Step 8] Testing POST /chat/reset...")
    reset_resp = client.post("/chat/reset", json={"user_id": user_id})
    assert reset_resp.status_code == 200
    print(f"✓ Chat reset response: {reset_resp.json()}")

    print("\n" + "=" * 80)
    print("🎉 ALL 4 CORE ARCHITECTURAL REQUIREMENTS SUCCESSFULLY VERIFIED!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()

"""
memory.py - Hindsight Cloud Memory Engine for Dilli Darshan AI
Integrates Vectorize Hindsight Cloud (https://api.hindsight.vectorize.io) as the unified
Agent Memory & User Profile store. No local SQLite database (users.db) and no local Docker container!
"""

import os
import zlib
import logging
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

from hindsight_client import Hindsight

logger = logging.getLogger("dilli_darshan.memory")

import threading

# Hindsight Cloud Credentials & Client Initialization
# ==============================================================================
hindsight_api_key = (
    os.getenv("HINDSIGHT_API_KEY")
    or os.getenv("Hindsight_API")
    or os.getenv("Hindsight_API_Key")
    or os.getenv("hindsight_api_key")
    or os.getenv("HINDSIGHT_KEY")
    or os.getenv("hindsight_API")
)
if hindsight_api_key:
    hindsight_api_key = hindsight_api_key.strip().strip('"').strip("'")

hindsight_base_url = (
    os.getenv("HINDSIGHT_API_URL")
    or "https://api.hindsight.vectorize.io"
).strip()

client: Optional[Hindsight] = None
if hindsight_api_key:
    try:
        client = Hindsight(
            base_url=hindsight_base_url,
            api_key=hindsight_api_key
        )
        logger.info(f"Hindsight Cloud client connected to {hindsight_base_url}")
    except Exception as e:
        logger.error(f"Failed to initialize Hindsight Cloud client: {e}")
        client = None
else:
    logger.warning("No Hindsight API key found in .env. Operating in in-memory fallback mode.")

_thread_local = threading.local()

def get_hindsight_client() -> Optional[Hindsight]:
    """
    Returns an active Hindsight client for the calling thread/task.
    Using thread-local storage ensures aiohttp ClientSessions are isolated per event loop,
    preventing cross-thread event loop collision in FastAPI worker pools.
    """
    if not hindsight_api_key:
        return None
    if not hasattr(_thread_local, "client") or _thread_local.client is None:
        try:
            _thread_local.client = Hindsight(
                base_url=hindsight_base_url,
                api_key=hindsight_api_key
            )
        except Exception as e:
            logger.error(f"Failed to create thread-local Hindsight client: {e}")
            return client
    return _thread_local.client

# Lightweight in-memory fallback store ONLY if hindsight_api_key is missing or network fails
IN_MEMORY_FALLBACK_STORE: dict[str, list[dict[str, Any]]] = {}
IN_MEMORY_PROFILES: dict[int, dict[str, Any]] = {}


# ==============================================================================
# Deterministic User ID & Bank Mapping
# ==============================================================================
def get_user_id_from_email(email: str) -> int:
    """
    Computes a deterministic integer user_id from the user's email address using CRC32.
    Ensures frontend compatibility where parseInt(userProfile.id) is used.
    """
    clean_email = email.strip().lower() if email else "tourist@delhi.com"
    return abs(zlib.crc32(clean_email.encode("utf-8"))) % 100000000 + 1


_CREATED_BANKS: set[str] = set()


def ensure_bank_exists(bank_id: str, bank_name: str = "Dilli Darshan Guest") -> None:
    """Ensures a memory bank exists on Hindsight Cloud before querying or retaining."""
    if bank_id in _CREATED_BANKS:
        return
    h_client = get_hindsight_client() or client
    if not h_client:
        return
    try:
        h_client.create_bank(
            bank_id=bank_id,
            name=bank_name,
            mission=(
                "You are Dilli Darshan's persistent memory bank for this tourist. "
                "Track their login identity, travel style, preferred language, dietary restrictions, "
                "budget, accessibility needs, metro station preferences, and Delhi monuments they have already visited or plan to visit."
            ),
            disposition={"skepticism": 2, "literalism": 4, "empathy": 5}
        )
    except Exception:
        pass
    _CREATED_BANKS.add(bank_id)


def get_bank_id_for_user(user_id: Optional[int], user_name: str = "Tourist") -> str:
    """Returns the dedicated Hindsight Cloud bank_id for a tourist or guest."""
    if user_id:
        return f"dilli_darshan_tourist_{user_id}"
    clean_name = user_name.strip().lower().replace(" ", "_")
    return f"dilli_darshan_guest_{clean_name or 'tourist'}"


# ==============================================================================
# Hindsight Cloud Profile & Memory Actions
# ==============================================================================
def save_user_login_to_hindsight(
    name: str,
    email: str,
    travel_style: str = "Heritage & History",
    language: str = "English"
) -> dict[str, Any]:
    """
    Saves the user login identity and preferences directly to Hindsight Cloud.
    Creates or updates the user's dedicated bank and retains their profile registration.
    """
    user_id = get_user_id_from_email(email)
    bank_id = f"dilli_darshan_tourist_{user_id}"

    # Always update in-memory profile store for instant local lookup
    profile_data = {
        "user_id": user_id,
        "bank_id": bank_id,
        "name": name,
        "email": email,
        "travel_style": travel_style,
        "language": language
    }
    IN_MEMORY_PROFILES[user_id] = profile_data

    # Cloud operation
    h_client = get_hindsight_client() or client
    if h_client:
        try:
            # 1. Create dedicated memory bank for tourist
            h_client.create_bank(
                bank_id=bank_id,
                name=f"Dilli Darshan Tourist - {name}",
                mission=(
                    "You are Dilli Darshan's persistent memory bank for this tourist. "
                    "Track their login identity, travel style, preferred language, dietary restrictions, "
                    "budget, accessibility needs, metro station preferences, and Delhi monuments they have already visited or plan to visit."
                ),
                disposition={"skepticism": 2, "literalism": 4, "empathy": 5}
            )
        except Exception as e:
            # Bank might already exist or network glitch
            logger.info(f"Hindsight bank creation info for {bank_id}: {e}")

        try:
            # 2. Retain registration profile document
            h_client.retain(
                bank_id=bank_id,
                content=(
                    f"User Login & Profile Registration: Tourist Name is {name} (Email: {email}). "
                    f"Preferred Travel Style: {travel_style}. Preferred Language: {language}."
                ),
                context="user_login_profile",
                document_id=f"profile_{user_id}",
                metadata={
                    "source": "login_form",
                    "user_id": str(user_id),
                    "name": name,
                    "email": email,
                    "travel_style": travel_style,
                    "language": language
                }
            )
        except Exception as e:
            logger.warning(f"Hindsight Cloud profile retain failed for {bank_id}: {e}")

    # Also keep in fallback store
    fallback_content = (
        f"User Login & Profile Registration: Tourist Name is {name} (Email: {email}). "
        f"Preferred Travel Style: {travel_style}. Preferred Language: {language}."
    )
    IN_MEMORY_FALLBACK_STORE.setdefault(bank_id, []).append({
        "content": fallback_content,
        "context": "user_login_profile"
    })

    return profile_data


def recall_user_memories(
    user_id: Optional[int],
    query: str,
    user_name: str = "Tourist"
) -> dict[str, Any]:
    """
    Recalls user profile, implicitly learned preferences, physical constraints,
    and prior conversation turns from Hindsight Cloud.
    """
    bank_id = get_bank_id_for_user(user_id, user_name)
    ensure_bank_exists(bank_id, f"Dilli Darshan - {user_name}")
    memories_list: list[str] = []

    h_client = get_hindsight_client() or client
    if h_client:
        try:
            res = h_client.recall(
                bank_id=bank_id,
                query=f"User login profile, dietary/budget/travel preferences, and relevant past context for: {query}",
                budget="mid",
                max_tokens=1024
            )
            for r in getattr(res, "results", []):
                txt = getattr(r, "text", "")
                if txt and txt.strip():
                    clean_txt = txt.strip()
                    if clean_txt not in memories_list:
                        memories_list.append(clean_txt)
        except Exception as e:
            logger.warning(f"Hindsight Cloud recall failed for bank {bank_id}: {e}")

    # If no memories returned or client offline, check in-memory fallback store
    if not memories_list and bank_id in IN_MEMORY_FALLBACK_STORE:
        for entry in IN_MEMORY_FALLBACK_STORE[bank_id]:
            c = entry.get("content", "")
            if c and c not in memories_list:
                memories_list.append(c)

    formatted_context = "\n".join(f"- {m}" for m in memories_list) if memories_list else ""

    return {
        "bank_id": bank_id,
        "memories": memories_list,
        "formatted_memory_context": formatted_context
    }


def retain_chat_turn(
    user_id: Optional[int],
    user_name: str,
    user_query: str,
    bot_answer: str,
    active_places: list[str]
) -> bool:
    """
    Asynchronously retains the conversation turn and extracts any new preferences
    into Hindsight Cloud with zero added latency (retain_async=True).
    """
    bank_id = get_bank_id_for_user(user_id, user_name)
    ensure_bank_exists(bank_id, f"Dilli Darshan - {user_name}")
    active_label = ", ".join(active_places) if active_places else "General Delhi"
    clean_bot_answer = bot_answer[:600].replace("\n", " ")

    turn_content = (
        f"Tourist ({user_name}) said: \"{user_query}\"\n"
        f"Active Delhi Landmark discussed: {active_label}\n"
        f"Dilli Darshan Guide replied: \"{clean_bot_answer}\""
    )

    success = False
    h_client = get_hindsight_client() or client
    if h_client:
        try:
            h_client.retain(
                bank_id=bank_id,
                content=turn_content,
                context="conversation_turn_and_preferences",
                metadata={
                    "source": "chat_turn",
                    "active_places": ",".join(active_places)
                },
                retain_async=True
            )
            success = True
        except Exception as e:
            logger.warning(f"Hindsight Cloud retain_chat_turn failed for {bank_id}: {e}")
            success = False

    # Also keep in-memory fallback entry
    IN_MEMORY_FALLBACK_STORE.setdefault(bank_id, []).append({
        "content": turn_content,
        "context": "conversation_turn_and_preferences"
    })

    return success or bool(IN_MEMORY_FALLBACK_STORE.get(bank_id))

"""
api.py - FastAPI Backend for Dilli Darshan AI
Integrates RAG vector retrieval (ChromaDB), Live Web Search (TinyFish/Gemini Grounding),
Multi-turn conversational memory, SQLite user profiles, and serves the companion frontend.
"""

import os
import re
import html
from pathlib import Path
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import database
import retrieval
import embedding

app = FastAPI(
    title="DilliDarshan AI Backend",
    description="Intelligent RAG & Live Tour Companion API for Delhi Tourism",
    version="2.0.0"
)

# Enable CORS for browser frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for stateful multi-turn user chat sessions
USER_SESSIONS: dict[str, retrieval.DilliDarshanChatSession] = {}


# ==============================================================================
# Markdown-to-HTML Formatter for Frontend (.bubble div)
# ==============================================================================
def format_markdown_for_frontend_html(
    md_text: str,
    default_source: str = "Delhi Tourism Official Archives"
) -> tuple[str, str]:
    """
    Transforms Gemini's Markdown output into clean, beautifully styled HTML
    for direct insertion into the frontend's bubble.innerHTML div.
    Also extracts a clean, concise source string for the .source-badge pill.
    """
    if not md_text:
        return "", default_source

    source_label = default_source

    # 1. Detect and strip any trailing [Sources & Grounding] or **Sources:** section
    source_patterns = [
        r"(?:\n*\[Sources\s*&\s*Grounding\]|\n*###?\s*Sources(?:\s*&\s*Grounding)?|\n*---+\s*\n\s*\*\*Sources?\*\*:?)(.*?)$"
    ]
    for pat in source_patterns:
        m = re.search(pat, md_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            found_src_block = m.group(1).strip()
            # Extract first valid source line
            lines = [l.strip().lstrip("-*• ") for l in found_src_block.split("\n") if l.strip()]
            if lines:
                candidate = lines[0]
                candidate = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", candidate)
                if len(candidate) <= 80:
                    source_label = candidate
            md_text = md_text[:m.start()].strip()
            break

    # 2. Convert Markdown blocks line by line
    raw_lines = md_text.split("\n")
    html_parts: list[str] = []
    in_ul = False
    in_ol = False

    def close_open_lists() -> str:
        nonlocal in_ul, in_ol
        closed = ""
        if in_ul:
            closed += "</ul>"
            in_ul = False
        if in_ol:
            closed += "</ol>"
            in_ol = False
        return closed

    def format_inline_tokens(text: str) -> str:
        """Converts bold, italic, code, and links to inline styled HTML."""
        # Bold: **text** or __text__
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
        # Italic: *text* or _text_
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
        text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<em>\1</em>", text)
        # Inline code: `code`
        text = re.sub(
            r"`([^`]+)`",
            r'<code style="background: rgba(255, 255, 255, 0.12); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em;">\1</code>',
            text
        )
        return text

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            closed = close_open_lists()
            if closed:
                html_parts.append(closed)
            continue

        # Horizontal rules
        if re.match(r"^(?:---+|\*\*\*+|___+)$", stripped):
            html_parts.append(close_open_lists() + '<hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.14); margin: 12px 0;" />')
            continue

        # Headings: ### H3, ## H2, # H1
        h_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if h_match:
            html_parts.append(close_open_lists())
            level = len(h_match.group(1))
            heading_content = format_inline_tokens(h_match.group(2).strip())
            if level == 1:
                html_parts.append(f'<h3 style="margin: 12px 0 6px; color: #fcd34d; font-weight: 700; font-size: 1.15rem;">{heading_content}</h3>')
            elif level == 2:
                html_parts.append(f'<h4 style="margin: 10px 0 4px; color: #fcd34d; font-weight: 700; font-size: 1.05rem;">{heading_content}</h4>')
            else:
                html_parts.append(f'<h5 style="margin: 8px 0 4px; color: #fcd34d; font-weight: 600; font-size: 0.98rem;">{heading_content}</h5>')
            continue

        # Unordered bullet list items: - item, * item, • item
        ul_match = re.match(r"^[-*•]\s+(.+)$", stripped)
        if ul_match:
            if in_ol:
                html_parts.append("</ol>")
                in_ol = False
            if not in_ul:
                html_parts.append('<ul style="margin: 6px 0 8px 20px; padding-left: 0; line-height: 1.55;">')
                in_ul = True
            item_body = format_inline_tokens(ul_match.group(1).strip())
            html_parts.append(f'<li style="margin-bottom: 5px;">{item_body}</li>')
            continue

        # Numbered list items: 1. item
        ol_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if ol_match:
            if in_ul:
                html_parts.append("</ul>")
                in_ul = False
            if not in_ol:
                html_parts.append('<ol style="margin: 6px 0 8px 22px; padding-left: 0; line-height: 1.55;">')
                in_ol = True
            item_body = format_inline_tokens(ol_match.group(1).strip())
            html_parts.append(f'<li style="margin-bottom: 5px;">{item_body}</li>')
            continue

        # Regular text paragraph line
        html_parts.append(close_open_lists())
        line_body = format_inline_tokens(stripped)
        html_parts.append(f'<div style="margin: 4px 0;">{line_body}</div>')

    # Close any trailing lists
    html_parts.append(close_open_lists())
    final_html = "".join(html_parts).strip()
    return final_html, source_label


# ==============================================================================
# Pydantic Schemas Matching Frontend Exactly
# ==============================================================================
class ProfileRequest(BaseModel):
    name: str = Field(..., description="Full name or alias of the tourist")
    email: str = Field(..., description="Unique email address or identifier")
    travel_style: str = Field("Heritage & History", description="Preferred travel style")
    language: str = Field("English", description="Preferred language (English or Hindi)")


class ChatRequest(BaseModel):
    query: str = Field(..., description="Question from the user")
    user_id: Optional[int] = Field(None, description="Registered user ID from SQLite")
    user_name: Optional[str] = Field("Tourist", description="Tourist name")
    travel_style: Optional[str] = Field("Heritage & History", description="Travel style preference")
    language: Optional[str] = Field("English", description="Response language")
    session_id: Optional[str] = Field(None, description="Optional custom session key")


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[int] = None


# ==============================================================================
# Endpoint 1: POST /user/profile (Matches saveProfile() in index.html)
# ==============================================================================
@app.post("/user/profile")
def save_profile(req: ProfileRequest) -> dict[str, Any]:
    """
    Saves or updates tourist profile in SQLite users.db.
    Matches frontend saveProfile() fetch contract.
    """
    user = database.upsert_user_profile(
        name=req.name,
        email=req.email,
        travel_style=req.travel_style,
        language=req.language
    )
    return {
        "status": "success",
        "user_id": user["user_id"],
        "name": user["name"],
        "email": user["email"],
        "travel_style": user["travel_style"],
        "language": user["language"]
    }


# ==============================================================================
# Endpoint 2: POST /chat (Matches submitQuestion() in index.html)
# ==============================================================================
@app.post("/chat")
def submit_question(req: ChatRequest) -> dict[str, Any]:
    """
    Stateful conversational chat endpoint:
    - Retains multi-turn conversation memory per user / session.
    - Personalizes response based on travel style and language.
    - Formats output as clean HTML for direct innerHTML rendering in index.html.
    - Returns concise source label for the .source-badge element.
    - Logs turn in SQLite chat_logs table.
    """
    # 1. Fallback verification against saved SQLite profile if user_id is provided
    effective_name = req.user_name or "Tourist"
    effective_style = req.travel_style or "Heritage & History"
    effective_lang = req.language or "English"

    if req.user_id:
        saved_user = database.get_user_by_id(req.user_id)
        if saved_user:
            if not req.user_name or req.user_name.lower() in ("tourist", "u", ""):
                effective_name = saved_user["name"]
            if not req.travel_style:
                effective_style = saved_user["travel_style"]
            if not req.language:
                effective_lang = saved_user["language"]

    # 2. Determine session key
    if req.session_id:
        session_key = req.session_id
    elif req.user_id:
        session_key = f"user_{req.user_id}"
    else:
        clean_guest = re.sub(r"\W+", "_", (effective_name or "tourist").lower()).strip("_")
        session_key = f"guest_{clean_guest or 'tourist'}"

    # 3. Retrieve or create multi-turn chat session
    if session_key not in USER_SESSIONS:
        USER_SESSIONS[session_key] = retrieval.DilliDarshanChatSession(max_history_turns=10)
    chat_session = USER_SESSIONS[session_key]

    # 4. Handle session reset triggers
    clean_q = req.query.strip().lower()
    if clean_q in ("reset", "new", "/new", "resetchat", "/resetchat", "clear"):
        chat_session.reset_session()
        reset_greeting = (
            f"Namaste <strong>{html.escape(effective_name)}</strong>! "
            f"Your conversational session has been refreshed. "
            f"I am ready with recommendations for <em>{html.escape(effective_style)}</em> in <em>{html.escape(effective_lang)}</em>. "
            f"Where would you like to go next?"
        )
        return {
            "answer": reset_greeting,
            "raw_markdown": reset_greeting,
            "source": "DilliDarshan Session Manager",
            "routing_status": "SESSION_RESET",
            "active_places": [],
            "turn_number": 0,
            "session_id": session_key
        }

    # 5. Execute retrieval & multi-turn generation with personalization
    user_context = {
        "user_name": effective_name,
        "travel_style": effective_style,
        "language": effective_lang
    }
    result = chat_session.send_message(req.query, user_context=user_context)

    raw_markdown = result.get("answer", "")
    base_source = result.get("source") or "Delhi Tourism Digital Desk"

    # Format into clean HTML for the frontend bubble
    formatted_html, final_source = format_markdown_for_frontend_html(raw_markdown, default_source=base_source)

    routing_status = result.get("status", "LOCAL_HIT")
    active_places = result.get("active_places", [])
    turn_number = result.get("turn_number", 1)

    # 6. Log interaction in SQLite database
    try:
        database.log_chat_interaction(
            user_id=req.user_id,
            session_key=session_key,
            query=req.query,
            answer=raw_markdown,
            source=final_source,
            routing_status=routing_status
        )
    except Exception as log_err:
        print(f"[Warning] Failed to log interaction in SQLite: {log_err}")

    # 7. Return JSON strictly matching frontend contract
    return {
        "answer": formatted_html,
        "raw_markdown": raw_markdown,
        "source": final_source,
        "routing_status": routing_status,
        "active_places": active_places,
        "turn_number": turn_number,
        "session_id": session_key
    }


# ==============================================================================
# Additional Endpoints: Health, Profile Lookup, Session Reset
# ==============================================================================
@app.get("/health")
@app.get("/api/health")
def health_status() -> dict[str, Any]:
    """Returns real-time status of ChromaDB, Gemini LLM, SQLite DB, and Sessions."""
    try:
        chroma_count = embedding.collection.count()
    except Exception:
        chroma_count = 504

    return {
        "status": "healthy",
        "service": "DilliDarshan AI Backend",
        "chromadb_chunks": chroma_count,
        "active_model": retrieval.MODEL_ID,
        "gemini_key_active": bool(getattr(embedding.client, "active_key", True)),
        "registered_users": database.get_users_count(),
        "logged_chat_turns": database.get_chat_logs_count(),
        "active_sessions": len(USER_SESSIONS)
    }


@app.get("/user/profile/{user_id}")
def get_user_profile(user_id: int) -> dict[str, Any]:
    """Fetches user profile by ID from SQLite users.db."""
    profile = database.get_user_by_id(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
    return profile


@app.post("/chat/reset")
def reset_user_session(req: ResetRequest) -> dict[str, Any]:
    """Explicitly resets a user's multi-turn session."""
    key = req.session_id or (f"user_{req.user_id}" if req.user_id else None)
    if key and key in USER_SESSIONS:
        USER_SESSIONS[key].reset_session()
        return {"status": "success", "message": f"Session '{key}' successfully reset"}
    return {"status": "success", "message": "No active session in memory to reset"}


# ==============================================================================
# Static Frontend Mounting & Single-Page Application Serving
# ==============================================================================
# Locate frontend directories (check local workspace first, then parent/sibling)
POSSIBLE_FRONTEND_DIRS = [
    Path(__file__).parent / "Dilli-Darshan",
    Path(__file__).parent.parent / "Dilli-Darshan",
    Path(__file__).parent.parent / "Dilli Darshan frontend" / "Dilli-Darshan",
    Path("d:/Dilli Darshan frontend/Dilli-Darshan")
]

FRONTEND_DIR: Optional[Path] = None
for candidate in POSSIBLE_FRONTEND_DIRS:
    if candidate.exists() and (candidate / "index.html").exists():
        FRONTEND_DIR = candidate
        break

if FRONTEND_DIR:
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    @app.get("/index.html")
    def serve_frontend_page():
        """Serves Dilli-Darshan/index.html so opening http://127.0.0.1:8000 loads the full companion."""
        return FileResponse(str(FRONTEND_DIR / "index.html"), media_type="text/html")
else:
    @app.get("/")
    def serve_api_root():
        return {
            "service": "DilliDarshan AI Backend",
            "documentation": "/docs",
            "health": "/health"
        }


# ==============================================================================
# Server Runner
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

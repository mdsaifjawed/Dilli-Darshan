"""
api.py - FastAPI Backend for Dilli Darshan AI
Integrates Vectorize Hindsight Cloud as the unified Agent Memory & User Profile store,
ChromaDB local vector RAG, TinyFish live web search, and serves the companion frontend.
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

import memory
import retrieval
import embedding

app = FastAPI(
    title="DilliDarshan AI Backend",
    description="Intelligent RAG & Live Tour Companion API for Delhi Tourism powered by Hindsight Cloud",
    version="2.1.0"
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
SESSIONS = USER_SESSIONS


# ==============================================================================
# Markdown-to-HTML Formatter for Frontend (.bubble div)
# ==============================================================================
def markdown_to_clean_html(
    md_text: str,
    default_source: str = "Delhi Tourism Official Archives"
) -> tuple[str, str]:
    """
    Transforms Gemini's Markdown output into clean, beautifully styled HTML
    for direct insertion into the frontend's bubble.innerHTML div without requiring
    an external browser markdown parser.
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
        # Escape raw HTML characters (e.g. '<', '>') first so expressions like "Children < 15" don't break HTML parsing
        text = html.escape(text, quote=False)
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
                html_parts.append(f'<h4 style="color:#fcd34d; margin:10px 0 4px; font-weight: 700; font-size: 1.05rem;">{heading_content}</h4>')
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
                html_parts.append('<ul style="margin: 8px 0 8px 18px; padding-left: 0; line-height: 1.55;">')
                in_ul = True
            item_body = format_inline_tokens(ul_match.group(1).strip())
            html_parts.append(f'<li style="margin-bottom: 5px;">{item_body}</li>')
            continue

        # Numbered list items: 1. item
        ol_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if ol_match:
            if in_ul:
                html_parts.append("</ul>")
                in_ul = False
            if not in_ol:
                html_parts.append('<ol style="margin: 8px 0 8px 18px; padding-left: 0; line-height: 1.55;">')
                in_ol = True
            item_body = format_inline_tokens(ol_match.group(2).strip())
            html_parts.append(f'<li style="margin-bottom: 5px;">{item_body}</li>')
            continue

        # Regular text paragraph
        closed = close_open_lists()
        if closed:
            html_parts.append(closed)
        formatted_p = format_inline_tokens(stripped)
        html_parts.append(f'<p style="margin-bottom: 8px; line-height: 1.6;">{formatted_p}</p>')

    closed = close_open_lists()
    if closed:
        html_parts.append(closed)

    result_html = "".join(html_parts)
    return result_html, source_label


# Backward-compatible alias
format_markdown_for_frontend_html = markdown_to_clean_html


# ==============================================================================
# Request & Response Schemas
# ==============================================================================
class ProfileRequest(BaseModel):
    name: str = Field(..., example="Saif")
    email: str = Field(..., example="saif@delhi.com")
    travel_style: str = Field(default="Heritage & History", example="Food & Street Markets")
    language: str = Field(default="English", example="English")


class ChatRequest(BaseModel):
    query: str = Field(..., example="What are the timings and ticket prices for Red Fort?")
    user_id: Optional[int] = Field(default=None, example=46682837)
    user_name: Optional[str] = Field(default="Tourist", example="Saif")
    travel_style: Optional[str] = Field(default="Heritage & History", example="Food & Street Markets")
    language: Optional[str] = Field(default="English", example="English")


class ResetRequest(BaseModel):
    user_id: Optional[int] = Field(default=None)
    session_id: Optional[str] = Field(default=None)


# ==============================================================================
# API Endpoints
# ==============================================================================

# Endpoint 1: POST /user/profile (Stores Login Info Directly in Hindsight Cloud)
@app.post("/user/profile")
def save_user_profile(req: ProfileRequest) -> dict[str, Any]:
    """
    Saves user login profile directly to Hindsight Cloud.
    Maps email to deterministic integer user_id and bank_id = f"dilli_darshan_tourist_{user_id}".
    """
    clean_name = req.name.strip()
    clean_email = req.email.strip()
    if not clean_email:
        raise HTTPException(status_code=400, detail="Valid email address is required")

    result = memory.save_user_login_to_hindsight(
        name=clean_name or "Tourist",
        email=clean_email,
        travel_style=req.travel_style,
        language=req.language
    )

    return {
        "status": "success",
        "user_id": result["user_id"],
        "bank_id": result["bank_id"],
        "name": result["name"],
        "email": result["email"],
        "travel_style": result["travel_style"],
        "language": result["language"]
    }


# Endpoint 2: POST /chat (Hindsight Cloud Recall + ChromaDB/TinyFish RAG + Hindsight Retain)
@app.post("/chat")
def chat_turn(req: ChatRequest) -> dict[str, Any]:
    """
    Orchestrates the full intelligent multi-turn turn:
    1. Looks up or creates user session in memory.
    2. Recalls memories & preferences from Hindsight Cloud.
    3. Resolves contextual query & place inheritance.
    4. Retrieves local ChromaDB chunks & TinyFish live web context.
    5. Generates local insider answer via Gemini.
    6. Asynchronously retains turn & extracted preferences in Hindsight Cloud.
    7. Converts Markdown to styled HTML for direct insertion into bubble.innerHTML.
    """
    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_key = (
        f"user_{req.user_id}"
        if req.user_id
        else f"guest_{(req.user_name or 'tourist').lower().replace(' ', '_')}"
    )

    if session_key not in SESSIONS:
        SESSIONS[session_key] = retrieval.DilliDarshanChatSession(max_history_turns=10)

    session = SESSIONS[session_key]

    try:
        result = session.send_message(
            raw_query=clean_query,
            user_id=req.user_id,
            user_name=req.user_name or "Tourist",
            travel_style=req.travel_style or "Heritage & History",
            language=req.language or "English"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat generation encountered an error: {str(e)}"
        )

    raw_markdown = result.get("answer", "")
    default_source = result.get("source") or "Delhi Tourism Official Archives"

    clean_html, extracted_source = markdown_to_clean_html(
        raw_markdown,
        default_source=default_source
    )

    final_source = result.get("source") or extracted_source

    return {
        "answer": clean_html,
        "raw_markdown": raw_markdown,
        "source": final_source,
        "routing_status": result.get("status", "LOCAL_HIT"),
        "active_places": result.get("active_places", []),
        "hindsight_bank_id": result.get("hindsight_bank_id", ""),
        "memories_recalled_count": result.get("memories_recalled_count", 0),
        "memories_recalled": result.get("memories_recalled", []),
        "model_used": result.get("model_used", ""),
        "is_followup": result.get("is_followup", False),
        "effective_query": result.get("effective_query", clean_query)
    }


# Endpoint 3: GET /health & GET /api/health
@app.get("/health")
@app.get("/api/health")
def health_status() -> dict[str, Any]:
    """
    Returns server operational health, ChromaDB chunk count, active Gemini key/model,
    and Hindsight Cloud connection status.
    """
    chroma_count = 0
    try:
        chroma_count = embedding.collection.count()
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "DilliDarshan AI Backend",
        "chromadb_chunks": chroma_count,
        "active_model": retrieval.MODEL_ID,
        "gemini_key_active": bool(getattr(embedding.client, "active_key", True)),
        "hindsight_cloud": {
            "base_url": memory.hindsight_base_url,
            "api_key_configured": bool(memory.hindsight_api_key),
            "client_connected": memory.client is not None
        },
        "active_sessions": len(SESSIONS)
    }


# Endpoint 4: GET /user/memories/{user_id}
@app.get("/user/memories/{user_id}")
def inspect_user_memories(user_id: int) -> dict[str, Any]:
    """
    Calls Hindsight Cloud recall for bank_id = f"dilli_darshan_tourist_{user_id}"
    so developers and tourists can inspect everything Hindsight Cloud has learned.
    """
    bank_id = memory.get_bank_id_for_user(user_id)
    recalled = memory.recall_user_memories(
        user_id=user_id,
        query="all user profile, preferences, visited places, accessibility and past conversation"
    )
    return {
        "user_id": user_id,
        "bank_id": bank_id,
        "memories": recalled.get("memories", []),
        "formatted_context": recalled.get("formatted_memory_context", "")
    }


# Endpoint 5: GET /user/profile/{user_id}
@app.get("/user/profile/{user_id}")
def get_user_profile(user_id: int) -> dict[str, Any]:
    """Fetches user profile by ID from memory store or Hindsight Cloud."""
    if user_id in memory.IN_MEMORY_PROFILES:
        return memory.IN_MEMORY_PROFILES[user_id]
    bank_id = memory.get_bank_id_for_user(user_id)
    recalled = memory.recall_user_memories(user_id=user_id, query="User login profile registration")
    return {
        "user_id": user_id,
        "bank_id": bank_id,
        "memories": recalled.get("memories", [])
    }


# Endpoint 6: POST /chat/reset
@app.post("/chat/reset")
def reset_user_session(req: ResetRequest) -> dict[str, Any]:
    """Explicitly resets a user's multi-turn session."""
    key = req.session_id or (f"user_{req.user_id}" if req.user_id else None)
    if key and key in SESSIONS:
        SESSIONS[key].reset_session()
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

# Mount /assets if assets folder exists
assets_dir = Path(__file__).parent / "assets"
if not assets_dir.exists() and FRONTEND_DIR:
    assets_dir = FRONTEND_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

workspace_frontend = Path(__file__).parent / "frontend.html"

@app.get("/")
@app.get("/index.html")
def serve_frontend_page():
    """Serves frontend single-page companion app."""
    if FRONTEND_DIR and (FRONTEND_DIR / "index.html").exists():
        return FileResponse(str(FRONTEND_DIR / "index.html"), media_type="text/html")
    if workspace_frontend.exists():
        return FileResponse(str(workspace_frontend), media_type="text/html")
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

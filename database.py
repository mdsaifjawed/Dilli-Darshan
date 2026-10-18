"""
database.py - Local SQLite User Profile & Chat Interaction Persistence for Dilli Darshan
Stores user preferences, travel styles, and persistent multi-turn chat logs.
Separate from vector database (chroma_db/chroma.sqlite3).
"""

import sqlite3
from pathlib import Path
from typing import Optional, Any

DB_PATH = Path(__file__).parent / "users.db"


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to users.db with row factory and WAL mode."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db() -> None:
    """
    Initializes SQLite tables if they do not exist:
    1. users: user profile, travel style, and language preferences.
    2. chat_logs: audit trail of queries, answers, grounding sources, and routing status.
    """
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    travel_style TEXT DEFAULT 'Heritage & History',
                    language TEXT DEFAULT 'English',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    session_key TEXT,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source TEXT,
                    routing_status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Index for case-insensitive email lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_email_lower 
                ON users(LOWER(email));
            """)
    finally:
        conn.close()


def upsert_user_profile(
    name: str,
    email: str,
    travel_style: str = "Heritage & History",
    language: str = "English"
) -> dict[str, Any]:
    """
    Upserts a user profile:
    - If email exists (case-insensitive), updates name, travel_style, language, and updated_at.
    - If not found, inserts a new user record.
    Returns: {"user_id": int, "name": str, "email": str, "travel_style": str, "language": str}
    """
    clean_name = name.strip() if name else "Tourist"
    clean_email = email.strip() if email else ""
    clean_style = travel_style.strip() if travel_style else "Heritage & History"
    clean_lang = language.strip() if language else "English"

    conn = get_db_connection()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM users WHERE LOWER(email) = LOWER(?) LIMIT 1;",
                (clean_email,)
            )
            row = cur.fetchone()

            if row:
                user_id = row["id"]
                cur.execute("""
                    UPDATE users 
                    SET name = ?, travel_style = ?, language = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?;
                """, (clean_name, clean_style, clean_lang, user_id))
            else:
                cur.execute("""
                    INSERT INTO users (name, email, travel_style, language)
                    VALUES (?, ?, ?, ?);
                """, (clean_name, clean_email, clean_style, clean_lang))
                user_id = cur.lastrowid

            return {
                "user_id": user_id,
                "name": clean_name,
                "email": clean_email,
                "travel_style": clean_style,
                "language": clean_lang
            }
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[dict[str, Any]]:
    """
    Fetches user profile by user_id. Returns dictionary or None if not found.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, email, travel_style, language, created_at, updated_at FROM users WHERE id = ? LIMIT 1;",
            (user_id,)
        )
        row = cur.fetchone()
        if not row:
            return None

        return {
            "user_id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "travel_style": row["travel_style"],
            "language": row["language"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"])
        }
    finally:
        conn.close()


def log_chat_interaction(
    user_id: Optional[int],
    session_key: str,
    query: str,
    answer: str,
    source: str,
    routing_status: str
) -> None:
    """
    Logs every chat interaction turn into the chat_logs table.
    """
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                INSERT INTO chat_logs (user_id, session_key, query, answer, source, routing_status)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (user_id, session_key, query, answer, source, routing_status))
    finally:
        conn.close()


def get_users_count() -> int:
    """Returns total count of registered users."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM users;")
        row = cur.fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def get_chat_logs_count() -> int:
    """Returns total count of logged interactions."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM chat_logs;")
        row = cur.fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


# Auto-initialize database on import
init_db()

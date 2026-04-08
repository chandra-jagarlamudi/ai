import os
import sqlite3
import uuid

DB_PATH = os.getenv("CHAT_DB_PATH", "data/chat_history.db")


def _connect() -> sqlite3.Connection:
    # Ensure the parent directory exists before SQLite tries to create the file.
    # os.path.abspath resolves relative paths (e.g. "data/chat_history.db" →
    # "/app/data/chat_history.db"), then dirname gives us "/app/data".
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                provider    TEXT NOT NULL,
                name        TEXT NOT NULL,
                model       TEXT NOT NULL DEFAULT '',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Migration: add `model` column to existing databases that predate this change.
        # SQLite raises OperationalError if the column already exists; we catch and
        # ignore it so this migration is safe to run on every startup.
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN model TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists


def create_session(provider: str, first_message: str, model: str = "") -> str:
    """Create a new session auto-named from the first user message."""
    session_id = str(uuid.uuid4())
    name = first_message[:50].strip()
    if len(first_message) > 50:
        name += "..."
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, provider, name, model) VALUES (?, ?, ?, ?)",
            (session_id, provider, name, model),
        )
    return session_id


def save_message(session_id: str, role: str, content: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )


def list_sessions(provider: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, model, created_at FROM sessions WHERE provider = ? ORDER BY updated_at DESC",
            (provider,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_messages(session_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str):
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

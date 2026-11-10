"""SQLite storage for user accounts (data/users.db)."""
import sqlite3
from contextlib import contextmanager

from werkzeug.security import generate_password_hash

from app.config import ADMIN_PASSWORD, ADMIN_USERNAME, USERS_DB_PATH


@contextmanager
def get_conn():
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the users table, and the initial admin account if there are no users yet."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (ADMIN_USERNAME, generate_password_hash(ADMIN_PASSWORD)),
            )
            print(f"Created initial user '{ADMIN_USERNAME}' (change its password after first login).")

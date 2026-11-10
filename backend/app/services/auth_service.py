"""Login and JWT issuing/verification."""
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from werkzeug.security import check_password_hash

from app.config import JWT_EXPIRES_MINUTES, JWT_SECRET
from app.db import get_conn

JWT_ALGORITHM = "HS256"
# Without a configured secret, tokens only stay valid until the backend restarts.
_SECRET = JWT_SECRET or secrets.token_hex(32)
if not JWT_SECRET:
    print("JWT_SECRET is not set; using a temporary one (logins reset when the backend restarts).")


class AuthError(Exception):
    """Raised for any auth failure (bad credentials, bad or expired token)."""


def authenticate(username: str, password: str) -> dict:
    """Verify credentials and return {"id", "username"}. Raises AuthError if invalid."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        raise AuthError("Invalid username or password.")
    return {"id": row["id"], "username": row["username"]}


def issue_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRES_MINUTES),
    }
    return jwt.encode(payload, _SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Verify a JWT and return {"id", "username"}. Raises AuthError if invalid/expired."""
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[JWT_ALGORITHM])
        return {"id": int(payload["sub"]), "username": payload["username"]}
    except (jwt.PyJWTError, KeyError, ValueError):
        raise AuthError("Your session has expired. Please log in again.")

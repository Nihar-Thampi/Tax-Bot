"""Flask decorator that enforces a valid JWT on protected routes."""
from functools import wraps

from flask import g, jsonify, request

from app.services.auth_service import AuthError, decode_token


def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Please log in."}), 401

        token = auth_header[len("Bearer "):].strip()
        try:
            g.user = decode_token(token)
        except AuthError as e:
            return jsonify({"error": str(e)}), 401

        return view(*args, **kwargs)

    return wrapped

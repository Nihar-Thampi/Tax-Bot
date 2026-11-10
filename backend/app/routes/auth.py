from flask import Blueprint, g, jsonify, request

from app.services.auth_service import AuthError, authenticate, issue_token
from app.utils.auth import require_auth

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    try:
        user = authenticate(username, password)
    except AuthError as e:
        return jsonify({"error": str(e)}), 401

    return jsonify({"token": issue_token(user), "user": {"username": user["username"]}})


@auth_bp.route("/auth/me", methods=["GET"])
@require_auth
def me():
    return jsonify({"user": {"username": g.user["username"]}})

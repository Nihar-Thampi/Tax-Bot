from flask import Blueprint, g, jsonify, request

from app.services.conversation_service import (
    ConversationError,
    delete_conversation,
    list_conversations,
    save_conversation,
)
from app.utils.auth import require_auth

conversations_bp = Blueprint("conversations", __name__)


@conversations_bp.route("/conversations", methods=["GET"])
@require_auth
def list_all():
    return jsonify({"conversations": list_conversations(g.user["username"])})


@conversations_bp.route("/conversations/<conversation_id>", methods=["PUT"])
@require_auth
def save(conversation_id):
    try:
        saved = save_conversation(g.user["username"], conversation_id, request.get_json() or {})
    except ConversationError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"conversation": saved})


@conversations_bp.route("/conversations/<conversation_id>", methods=["DELETE"])
@require_auth
def delete(conversation_id):
    try:
        delete_conversation(g.user["username"], conversation_id)
    except ConversationError as e:
        return jsonify({"error": str(e)}), 400
    return "", 204

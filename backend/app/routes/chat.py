import logging

from flask import Blueprint, g, jsonify, request

from app.services.cancellation import GenerationCancelled, cancel, cancellable
from app.services.chat_service import get_response_with_citations
from app.utils.auth import require_auth

chat_bp = Blueprint("chat", __name__)
logger = logging.getLogger(__name__)


@chat_bp.route("/chat", methods=["POST"])
@require_auth
def chat():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []
    # Client-chosen id so POST /api/chat/cancel can stop this request.
    request_id = data.get("request_id") or None
    if not isinstance(history, list):
        history = []

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        # use_finetuned=None: RAG over the Act unless TAX_MODEL_ADAPTER points at a trained adapter
        # (same rule /api/health reports).
        with cancellable(g.user["username"], request_id):
            reply, new_history, citations = get_response_with_citations(message, history, use_finetuned=None)
        return jsonify({
            "reply": reply,
            "history": new_history,
            "points": citations["points"],
            "sources": citations["sources"],
        })
    except GenerationCancelled:
        return jsonify({"cancelled": True, "error": "Request cancelled by user."}), 409
    except Exception:
        # Full traceback in the backend log; the user only sees a generic message.
        logger.exception("Chat request failed")
        return jsonify({"error": "There was an issue generating an answer."}), 500


@chat_bp.route("/chat/cancel", methods=["POST"])
@require_auth
def cancel_chat():
    request_id = ((request.get_json() or {}).get("request_id") or "").strip()
    if not request_id:
        return jsonify({"error": "request_id is required"}), 400
    return jsonify({"cancelled": cancel(g.user["username"], request_id)})

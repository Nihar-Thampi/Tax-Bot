from flask import Blueprint, jsonify, request

from app.services.chat_service import get_response

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []
    if not isinstance(history, list):
        history = []

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        reply, new_history = get_response(message, history, use_finetuned=True)
        return jsonify({"reply": reply, "history": new_history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

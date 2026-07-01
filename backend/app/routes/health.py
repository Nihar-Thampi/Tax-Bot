from flask import Blueprint, jsonify

from app.config import use_fine_tuned_model

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "use_finetuned": use_fine_tuned_model()})

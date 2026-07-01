"""
Flask UI for the fine-tuned SA tax chatbot.
Run: set TAX_MODEL_ADAPTER=tax_lora_adapter  (or your adapter path), then python app.py
"""
from flask import Flask, render_template, request, jsonify

from tax_qa_chatbot import get_response, _use_fine_tuned_model

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", use_finetuned=_use_fine_tuned_model())


@app.route("/api/chat", methods=["POST"])
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


if __name__ == "__main__":
    print("SA Tax Chatbot (fine-tuned) - open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)

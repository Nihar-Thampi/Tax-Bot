"""
Interactive terminal chat session against the SA tax chatbot (RAG or fine-tuned).

Usage (run from backend/):
  python scripts/chat_cli.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import use_fine_tuned_model
from app.services.chat_service import get_response, load_history, save_history


def chat_loop() -> None:
    """Run an interactive chat session with persisted history."""
    history = load_history()
    if history:
        print(f"  (Loaded {len(history)} previous messages from history.)")

    use_ft = use_fine_tuned_model()
    print("=" * 60)
    print("  SA Income Tax Act -- Q&A Chatbot (local Hugging Face LLM)")
    if use_ft:
        print("  Mode: fine-tuned model (answers from trained knowledge of the Act)")
    else:
        print("  Mode: RAG (answers from retrieved excerpts). Set TAX_MODEL_ADAPTER to use fine-tuned model.")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            save_history(history)
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            save_history(history)
            print("Goodbye!")
            break

        try:
            answer, history = get_response(user_input, history, use_finetuned=use_ft)
            save_history(history)
            print(f"\nAssistant: {answer}")
        except Exception as exc:
            print(f"\nError: {exc}")
            print("Please try again.")


def main() -> None:
    chat_loop()


if __name__ == "__main__":
    main()

import json
import os
import textwrap
from pathlib import Path

from env_config import get_env
from tax_law_rag import retrieve
from local_llm import generate

MAX_HISTORY = 10
HISTORY_FILE = Path(__file__).resolve().parent / "chat_history.json"


def _use_fine_tuned_model() -> bool:
    """True if a fine-tuned adapter is loaded (model was trained on the Act)."""
    path = get_env("TAX_MODEL_ADAPTER")
    return bool(path and os.path.isdir(path))


SYSTEM_PROMPT_RAG = textwrap.dedent("""\
    You are a South African tax law assistant. Answer using ONLY the CONTEXT from the Act provided below.

    CRITICAL: Write your answer in your own words. Do NOT copy, paste, or continue any list or paragraph from the context.
    Do NOT output text that looks like "(1) companies... (2) companies..." or "section sixty-X" repeated lines.
    Give a short summary (2-4 paragraphs) and end with "Net result:" and one sentence takeaway.
    If the context does not cover the question, say so. Cite section numbers when relevant (e.g. Section 11(a)).
    End with a one-line disclaimer that this is not professional advice; consult a tax practitioner.
""")


SYSTEM_PROMPT_FINETUNED = textwrap.dedent("""\
    You are a South African tax law assistant. You were trained on the Income Tax Act No. 58 of 1962.
    Answer the user's question from your knowledge of the Act. Be concise: 2-4 short paragraphs, then "Net result:" and one sentence.
    Cite section numbers when relevant (e.g. Section 11(a), Section 18). Do not copy long lists.
    End with a one-line disclaimer: this is not professional advice; consult a registered tax practitioner.
""")


def load_history() -> list[dict]:
    """Load chat history from file if it exists."""
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data[-MAX_HISTORY * 2:]
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _trim_list_dump(text: str, finetuned: bool = False) -> str:
    """If the model echoed a long list from the Act, truncate and add a note.
    When finetuned=True, require several consecutive list lines before cutting so short answers (e.g. starting with '18. Deduction...') are kept."""
    import re
    lines = text.strip().split("\n")
    list_line = re.compile(r"^\s*\(\d+\)\s+.+", re.I)
    section_line = re.compile(r"section\s+sixty[- ]?(one|two|three|\d+)", re.I)
    numbered_dot = re.compile(r"^\s*\d+\.\s+", re.I)
    roman_paren = re.compile(r"\(\s*x{0,3}i{0,3}v?i{0,3}x*\s*\)", re.I)
    deemed_phrase = re.compile(
        r"shall be deemed.*(?:subject to a deduction|provisions of this section)",
        re.I | re.DOTALL,
    )
    company_deemed_dump = re.compile(
        r"any company which.*subject to a deduction.*shall be deemed",
        re.I | re.DOTALL,
    )
    min_list_lines = 4 if finetuned else 1
    count = 0
    cut = len(lines)
    for i, line in enumerate(lines):
        if company_deemed_dump.search(line):
            cut = i
            break
        stripped = line.strip()
        is_list = (
            list_line.match(stripped)
            or section_line.search(line)
            or numbered_dot.match(stripped)
            or roman_paren.search(line)
            or deemed_phrase.search(line)
        )
        if is_list:
            count += 1
            if count >= min_list_lines:
                cut = i - min_list_lines + 1
                break
        else:
            count = 0
    if cut < len(lines):
        kept = "\n".join(lines[:cut]).strip()
        if len(kept) < 80 or (kept and kept[0].islower()):
            if finetuned:
                return (
                    "The model did not give a clear short answer. Try rephrasing your question, "
                    "or train on more data (python train_tax_lora.py --max-examples 0) for better coverage. "
                    "This is not professional tax advice; consult a registered tax practitioner."
                )
            return (
                "The retrieved context did not contain a clear short answer for this question, "
                "or the model repeated a long passage from the Act. Try asking more specifically. "
                "This is not professional tax advice; consult a registered tax practitioner."
            )
        return kept + "\n\n[Answer shortened; the context contained a long list not relevant to a short summary.]"
    return text


def save_history(history: list[dict]) -> None:
    """Persist chat history to file (keep last MAX_HISTORY*2 messages)."""
    try:
        to_save = history[-(MAX_HISTORY * 2):]
        HISTORY_FILE.write_text(
            json.dumps(to_save, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def get_response(
    user_input: str,
    history: list[dict],
    use_finetuned: bool | None = None,
) -> tuple[str, list[dict]]:
    """Get one assistant reply and updated history. use_finetuned=None means auto from TAX_MODEL_ADAPTER."""
    use_ft = use_finetuned if use_finetuned is not None else _use_fine_tuned_model()
    if use_ft:
        system_prompt = SYSTEM_PROMPT_FINETUNED
        user_content = user_input.strip()
    else:
        system_prompt = SYSTEM_PROMPT_RAG
        context_chunks = retrieve(user_input.strip(), n_results=8)
        context_block = "\n\n---\n\n".join(
            f"[CONTEXT {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        user_content = (
            f"CONTEXT FROM THE ACT:\n{context_block}\n\n"
            f"USER QUESTION: {user_input.strip()}\n\n"
            "Answer the question in 2-4 short paragraphs in your own words. Do not copy or continue any list from the context. End with 'Net result:' and one sentence."
        )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-MAX_HISTORY * 2:])
    messages.append({"role": "user", "content": user_content})

    answer = generate(messages, max_tokens=1024, temperature=0.3)
    if use_ft and len(answer) > 400:
        answer = _trim_list_dump(answer, finetuned=True)
    elif not use_ft:
        answer = _trim_list_dump(answer, finetuned=False)

    new_history = list(history) + [
        {"role": "user", "content": user_input.strip()},
        {"role": "assistant", "content": answer},
    ]
    return answer, new_history


def chat_loop() -> None:
    """Run an interactive chat session with persisted history."""
    history = load_history()
    if history:
        print(f"  (Loaded {len(history)} previous messages from history.)")

    use_ft = _use_fine_tuned_model()
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

        if use_ft:
            system_prompt = SYSTEM_PROMPT_FINETUNED
            user_content = user_input
        else:
            system_prompt = SYSTEM_PROMPT_RAG
            context_chunks = retrieve(user_input, n_results=8)
            context_block = "\n\n---\n\n".join(
                f"[CONTEXT {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
            )
            user_content = (
                f"CONTEXT FROM THE ACT:\n{context_block}\n\n"
                f"USER QUESTION: {user_input}\n\n"
                "Answer the question in 2-4 short paragraphs in your own words. Do not copy or continue any list from the context. End with 'Net result:' and one sentence."
            )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-MAX_HISTORY * 2:])
        messages.append({"role": "user", "content": user_content})

        try:
            answer = generate(
                messages,
                max_tokens=1024,
                temperature=0.3,
            )
            if use_ft and len(answer) > 400:
                answer = _trim_list_dump(answer, finetuned=True)
            elif not use_ft:
                answer = _trim_list_dump(answer, finetuned=False)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})
            save_history(history)

            print(f"\nAssistant: {answer}")

        except Exception as exc:
            print(f"\nError: {exc}")
            print("Please try again.")


def main() -> None:
    chat_loop()


if __name__ == "__main__":
    main()

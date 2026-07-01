import json
import re
import textwrap

from app.config import CHAT_HISTORY_PATH, use_fine_tuned_model
from app.services.llm_service import generate
from app.services.rag_service import retrieve

MAX_HISTORY = 10

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
    if not CHAT_HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data[-MAX_HISTORY * 2:]
        return []
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history: list[dict]) -> None:
    """Persist chat history to file (keep last MAX_HISTORY*2 messages)."""
    try:
        to_save = history[-(MAX_HISTORY * 2):]
        CHAT_HISTORY_PATH.write_text(
            json.dumps(to_save, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _trim_list_dump(text: str, finetuned: bool = False) -> str:
    """If the model echoed a long list from the Act, truncate and add a note.
    When finetuned=True, require several consecutive list lines before cutting so short answers (e.g. starting with '18. Deduction...') are kept."""
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


def get_response(
    user_input: str,
    history: list[dict],
    use_finetuned: bool | None = None,
) -> tuple[str, list[dict]]:
    """Get one assistant reply and updated history. use_finetuned=None means auto from TAX_MODEL_ADAPTER."""
    use_ft = use_finetuned if use_finetuned is not None else use_fine_tuned_model()
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

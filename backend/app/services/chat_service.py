import json
import re
import textwrap

from app.config import CHAT_HISTORY_PATH, use_fine_tuned_model
from app.services.llm_service import (
    context_window,
    count_tokens,
    generate,
    text_tokens,
    truncate_text_tokens,
)
from app.services.citation_service import cite_answer, is_contents_chunk, section_text, sections_for_question
from app.services.rag_service import retrieve

MAX_HISTORY = 10
# Earlier turns sent with each question (user + assistant messages). Act excerpts fill the
# space left after the question and history, so a small-context model just gets fewer.
PROMPT_HISTORY_MESSAGES = 6
RAG_N_RESULTS = 5
# Of those, up to this many are whole sections picked by title; the rest come from the vector index.
TITLE_SECTIONS = 2
# Cap per excerpt so one long section (e.g. 10, Exemptions) can't crowd out the rest.
MAX_EXCERPT_TOKENS = 1600
CHAT_MAX_NEW_TOKENS = 900
PROMPT_SAFETY_MARGIN = 16
NO_CITED_ANSWER = (
    "I couldn't find a passage in the Income Tax Act that supports an answer to this. "
    "Try rephrasing the question or asking about a specific section."
)

SYSTEM_PROMPT_RAG = textwrap.dedent("""\
    You are a South African tax assistant who explains the Income Tax Act No. 58 of 1962 to
    people who are not tax experts. You are given excerpts from the Act; base everything you say on them.

    How to answer:
    - Start with a direct answer to the question in one or two sentences.
    - Then explain each relevant provision as its own bullet starting with "- ". In each bullet say,
      in plain language, what the provision allows or requires, the conditions, who it applies to,
      and what it means in practice (a short example helps). Name the section, e.g. "(Section 18)".
    - Only explain provisions that actually answer the question. Some excerpts will be unrelated;
      skip those. Never name a section that does not appear in the excerpts.
    - Do not quote the Act at length; explain it.
    - If the excerpts do not cover part of the question, say so plainly instead of guessing.
    - Plain text only: no bold, headings or tables.

    The excerpts are a scan of the Act as published in 1962, so they contain OCR errors; read past them.
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


def _strip_markdown(text: str) -> str:
    """The UI shows plain text, so drop bold/heading markers the model may still add."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    return re.sub(r"^\s*[*•]\s+", "- ", text, flags=re.M).strip()


def _retrieve_provisions(question: str, n: int) -> list[str]:
    """Act text for a question, most relevant first.

    Sections whose table-of-contents title matches the question come first, as whole
    sections (titles are clean text, so this is far more reliable than searching the OCR'd
    body). The vector index then fills the rest, skipping the table of contents itself
    (it matches many queries on its titles alone but holds no provisions to answer from).
    """
    by_title = [text for num in sections_for_question(question, k=TITLE_SECTIONS)
                if (text := section_text(num))]
    by_vector = [c for c in retrieve(question, n_results=n + 2) if not is_contents_chunk(c)]
    out: list[str] = []
    for chunk in by_title + by_vector:
        # Skip vector hits already covered by a whole section.
        if not any(chunk[:200] in seen for seen in out):
            out.append(chunk)
    return out[:n]


def get_response_with_citations(
    user_input: str,
    history: list[dict],
    use_finetuned: bool | None = None,
) -> tuple[str, list[dict], dict]:
    """Like get_response, plus {"points", "sources"} tying each point of the answer to the Act."""
    use_ft = use_finetuned if use_finetuned is not None else use_fine_tuned_model()
    question = user_input.strip()
    prompt_history = history[-PROMPT_HISTORY_MESSAGES:] if PROMPT_HISTORY_MESSAGES else []
    input_budget = context_window() - CHAT_MAX_NEW_TOKENS - PROMPT_SAFETY_MARGIN

    def build(user_content: str, hist: list[dict]) -> list[dict]:
        system_prompt = SYSTEM_PROMPT_FINETUNED if use_ft else SYSTEM_PROMPT_RAG
        return [{"role": "system", "content": system_prompt}, *hist,
                {"role": "user", "content": user_content}]

    def rag_user_content(context_block: str) -> str:
        return (
            f"EXCERPTS FROM THE ACT:\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            "Explain the answer in plain language using the excerpts, following the format above."
        )

    if use_ft:
        user_content = question
    else:
        user_content = rag_user_content("")

    # Drop history if even the bare question doesn't leave room.
    if count_tokens(build(user_content, prompt_history)) > input_budget:
        prompt_history = []

    # Act text the answer is checked against for citations: what the model was shown in
    # RAG mode, or the closest passages to the question in fine-tuned mode.
    cited_chunks: list[str] = []
    if use_ft:
        cited_chunks = _retrieve_provisions(question, RAG_N_RESULTS * 2)
    else:
        # Fill the remaining space with Act excerpts, best match first.
        remaining = input_budget - count_tokens(build(user_content, prompt_history))
        parts: list[str] = []
        for i, chunk in enumerate(_retrieve_provisions(question, RAG_N_RESULTS)):
            chunk = truncate_text_tokens(chunk, MAX_EXCERPT_TOKENS)
            label = f"[EXCERPT {i+1}]\n"
            cost = text_tokens(label + chunk) + 8  # separator + slack
            if cost > remaining:
                if remaining > 64:
                    chunk = truncate_text_tokens(chunk, remaining - 8 - text_tokens(label))
                    parts.append(label + chunk)
                    cited_chunks.append(chunk)
                break
            parts.append(label + chunk)
            cited_chunks.append(chunk)
            remaining -= cost
        user_content = rag_user_content("\n\n---\n\n".join(parts))

    messages = build(user_content, prompt_history)

    answer = generate(
        messages,
        max_tokens=CHAT_MAX_NEW_TOKENS,
        temperature=0.3,
        repetition_penalty=1.05,
    )
    # The fine-tuned (TinyLlama) adapter tends to dump lists from the Act; trim those.
    if use_ft and len(answer) > 400:
        answer = _trim_list_dump(answer, finetuned=True)
    answer = _strip_markdown(answer)

    # Only show points that could be tied to a passage of the Act; drop the rest.
    citations = cite_answer(answer, cited_chunks)
    shown = [p for p in citations["points"] if p["kind"] != "unsupported"]
    if not any(p["kind"] == "cited" for p in shown):
        shown = [{"text": NO_CITED_ANSWER, "source_id": None, "score": None, "kind": "disclaimer"}]
    citations["points"] = shown
    answer = "\n\n".join(p["text"] for p in shown)

    new_history = list(history) + [
        {"role": "user", "content": user_input.strip()},
        {"role": "assistant", "content": answer},
    ]
    return answer, new_history, citations


def get_response(
    user_input: str,
    history: list[dict],
    use_finetuned: bool | None = None,
) -> tuple[str, list[dict]]:
    """Get one assistant reply and updated history. use_finetuned=None means auto from TAX_MODEL_ADAPTER."""
    answer, new_history, _ = get_response_with_citations(user_input, history, use_finetuned)
    return answer, new_history

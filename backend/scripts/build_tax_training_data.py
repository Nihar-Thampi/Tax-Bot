"""
Build instruction-tuning dataset from the SA Income Tax Act.
Each example: (instruction, Act excerpt, short summary). The model is trained to
produce the summary from the excerpt so it learns the law and can infer answers
without relying only on RAG retrieval.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import LAW_TEXT_PATH, TRAINING_DATA_PATH as OUT_PATH

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > CHUNK_SIZE and current:
            chunks.append(current.strip())
            words = current.split()
            overlap_words = words[-CHUNK_OVERLAP // 6:] if len(words) > CHUNK_OVERLAP // 6 else []
            current = " ".join(overlap_words) + "\n\n" + para
        else:
            current = (current + "\n\n" + para) if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _first_sentences(text: str, max_chars: int = 280) -> str:
    """First 1-3 sentences or up to max_chars, for use as summary target."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    end = text.find(".", 0)
    if end == -1:
        return text[:max_chars].rsplit(" ", 1)[0] + "."
    last = end + 1
    while last < len(text) and last < max_chars:
        next_dot = text.find(".", last)
        if next_dot == -1:
            break
        last = next_dot + 1
    out = text[:last].strip()
    return out if out else text[:max_chars].rsplit(" ", 1)[0] + "."


def _is_repetitive_list(chunk: str) -> bool:
    """True if chunk is mostly the 'Any company... shall be deemed...' style list we don't want to teach."""
    pattern = re.compile(
        r"shall be deemed to be a company which is subject to a deduction",
        re.I,
    )
    return len(pattern.findall(chunk)) >= 2


def build_dataset(
    law_path: Path | None = None,
    out_path: Path | None = None,
    max_chunks: int | None = None,
) -> None:
    law_path = law_path or LAW_TEXT_PATH
    out_path = out_path or OUT_PATH
    print(f"Reading {law_path} ...")
    raw = law_path.read_text(encoding="utf-8")
    cleaned = _clean(raw)
    chunks = _chunk_text(cleaned)
    if max_chunks and max_chunks > 0 and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        print(f"Using first {max_chunks} chunks (light dataset for fast training)")
    else:
        print(f"Created {len(chunks)} chunks")

    instruction = (
        "You are an SA tax assistant. Based only on the following excerpt from the "
        "Income Tax Act No. 58 of 1962, give a short answer in 2-4 sentences. "
        "Do not copy long lists. State the main rule or deduction and end with a one-line net result."
    )

    count = 0
    skipped_repetitive = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            if _is_repetitive_list(chunk):
                skipped_repetitive += 1
                continue
            summary = _first_sentences(chunk)
            if len(summary) < 30:
                continue
            if _is_repetitive_list(summary):
                summary = "This part of the Act concerns companies and deduction provisions. Do not copy long lists."
            record = {
                "instruction": instruction,
                "input": f"Excerpt from the Act:\n\n{chunk}",
                "output": summary,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if skipped_repetitive:
        print(f"Skipped {skipped_repetitive} repetitive list chunks.")
    print(f"Wrote {count} examples to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build tax law instruction dataset from the Act.")
    p.add_argument("-o", "--output", type=Path, default=OUT_PATH, help="Output JSONL path.")
    p.add_argument("-f", "--file", type=Path, default=LAW_TEXT_PATH, help="Path to Act text file.")
    p.add_argument("--max-chunks", type=int, default=None, help="Use only first N chunks (for fast/light training).")
    args = p.parse_args()
    build_dataset(law_path=args.file, out_path=args.output, max_chunks=args.max_chunks)

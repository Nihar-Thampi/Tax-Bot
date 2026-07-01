"""
Pure document training: extract clean text from Income Tax Act and optional guides.
Output: one corpus file (~500k-2M tokens) for causal LM training (no Q&A).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CORPUS_PATH, DATA_DIR, LAW_TEXT_PATH as ACT_PATH

# Optional: add paths to PDFs when you have them (SARS Budget 2026, Small Business Guide)
EXTRA_PATHS = [
    DATA_DIR / "sars_budget_2026_tax_guide.pdf",
    DATA_DIR / "tax_guide_small_businesses.pdf",
]


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n\n".join(parts)
    except Exception as e:
        print(f"  Skip {path.name}: {e}")
        return ""


def build_corpus(
    act_path: Path | None = None,
    extra_paths: list[Path] | None = None,
    out_path: Path | None = None,
) -> None:
    act_path = act_path or ACT_PATH
    extra_paths = extra_paths or EXTRA_PATHS
    out_path = out_path or CORPUS_PATH

    parts = []

    if act_path.exists():
        print(f"Reading Act: {act_path.name}")
        raw = act_path.read_text(encoding="utf-8")
        parts.append(_clean(raw))
    else:
        print(f"Act not found: {act_path}")

    for p in extra_paths:
        if not p.exists():
            continue
        if p.suffix.lower() == ".pdf":
            print(f"Reading PDF: {p.name}")
            parts.append(_clean(_read_pdf(p)))
        elif p.suffix.lower() in (".txt", ".md"):
            print(f"Reading text: {p.name}")
            parts.append(_clean(p.read_text(encoding="utf-8")))

    if not parts:
        print("No input files found. Add za-act-1962-58-publication-document.txt or PDFs.")
        return

    corpus = "\n\n---\n\n".join(parts)
    out_path.write_text(corpus, encoding="utf-8")
    approx_tokens = len(corpus.split()) * 4 // 3
    print(f"Wrote {len(corpus):,} chars (~{approx_tokens:,} tokens) to {out_path.name}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build raw text corpus for causal LM training.")
    p.add_argument("-o", "--output", type=Path, default=CORPUS_PATH, help="Output corpus path.")
    p.add_argument("-f", "--file", type=Path, action="append", help="Extra file (PDF or TXT). Can repeat.")
    args = p.parse_args()
    build_corpus(out_path=args.output, extra_paths=args.file or EXTRA_PATHS)

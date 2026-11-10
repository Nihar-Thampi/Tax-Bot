"""Attach a source from the Act to each point of an answer.

The Act text is split into paragraphs the same way the RAG index chunks it, and every
line is tagged with the section it falls under, its line number in the .txt file and
its Government Gazette page. Each point of the answer is then matched by embedding
similarity to the pieces of the Act that were actually retrieved for the prompt. A
point with no close match is reported as unsupported rather than given a made-up citation.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import EMBEDDING_MODEL, LAW_TEXT_PATH, get_env

# Cosine similarity (all-MiniLM-L6-v2) needed before a piece of the Act counts as a point's source.
MIN_SIMILARITY = float(get_env("CITATION_MIN_SIMILARITY", "0.45"))
MIN_UNIT_CHARS = 40
EXCERPT_CHARS = 600
# A Gazette page is ~85 lines of the OCR text; headers more than this far off the expected page are OCR errors.
PAGE_TOLERANCE = 6
PAGE_MAX_LINES = 100
PREAMBLE = "Preamble / contents"
DISCLAIMER_MAX_CHARS = 250

# Body of the Act starts at "1. In this Act ..."; everything before is the table of contents.
_BODY_START = re.compile(r"^\s*1\.\s+In this Act\b")
# Section headings as OCR'd: "11." can come out as "ll.", "80." as "&0.", "81." as "$1."
_SECTION_START = re.compile(r"^\s*([\dlI|O&$S]{1,3}[A-Z]?)[.,]\s+(?:\((\d+)\)\s*)?\S")
_OCR_DIGITS = str.maketrans({"l": "1", "I": "1", "|": "1", "O": "0", "&": "8", "$": "8", "S": "5"})
_SUBSECTION_START = re.compile(r"^\s*\((\d+)\)\s+\S")
_SCHEDULE = re.compile(r"^\s*(First|Second|Third|Fourth|Fifth)\s+Schedule\.?\s*$", re.I)
_GAZETTE_PAGE = re.compile(r"GAZETTE.*?(\d{1,3})\s*\|?\s*$", re.I)
_DISCLAIMER = re.compile(
    r"not (?:constitute )?professional|consult a (?:registered )?tax practitioner|^\[Answer shortened", re.I
)
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


def _clean(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", text).strip()


@dataclass
class Unit:
    """A run of lines inside one paragraph that all fall under the same section."""
    text: str
    line_start: int
    line_end: int
    section: str
    page: int | None


@dataclass
class Line:
    number: int
    text: str
    section: str
    page: int | None


@dataclass
class Paragraph:
    text: str
    lines: list[Line]

    def units(self) -> list[Unit]:
        runs: list[list[Line]] = []
        for line in self.lines:
            if runs and runs[-1][-1].section == line.section:
                runs[-1].append(line)
            else:
                runs.append([line])
        return [
            Unit(_clean("\n".join(l.text for l in run)), run[0].number, run[-1].number,
                 run[0].section, run[0].page)
            for run in runs
        ]


def _trusted_page_headers(raw_lines: list[str]) -> dict[int, int]:
    """Line number -> Gazette page, keeping only headers whose OCR'd number is plausible.

    Page numbers grow roughly linearly with line number, so headers far off a
    least-squares line (OCR misreads like "31" for "81") are dropped.
    """
    import numpy as np
    found = [(n, int(m.group(1))) for n, line in enumerate(raw_lines, start=1)
             if (m := _GAZETTE_PAGE.search(line))]
    if len(found) < 3:
        return dict(found)
    keep = found
    for _ in range(3):
        xs = np.array([n for n, _ in keep], dtype=float)
        ys = np.array([p for _, p in keep], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        keep = [(n, p) for n, p in found if abs(p - (slope * n + intercept)) <= PAGE_TOLERANCE]
    return dict(keep)


@lru_cache(maxsize=1)
def _paragraph_index() -> dict[str, Paragraph]:
    """Cleaned paragraph text (as the RAG chunker produces it) -> its lines in the Act."""
    raw_lines = LAW_TEXT_PATH.read_text(encoding="utf-8").split("\n")
    page_headers = _trusted_page_headers(raw_lines)
    index: dict[str, Paragraph] = {}

    in_body = False
    schedule = ""
    section = PREAMBLE
    section_no = 0
    subsection_no = 0
    header_page: int | None = None
    header_line = 0
    section_start_line = 0
    # (first line, line after last, from section, to section) where a heading was unreadable
    gaps: list[tuple[int, int, int, int]] = []
    buf: list[Line] = []

    def flush() -> None:
        if buf:
            text = _clean("\n".join(l.text for l in buf))
            if text:
                index.setdefault(text, Paragraph(text, list(buf)))
            buf.clear()

    for n, line in enumerate(raw_lines, start=1):
        if line == "":
            flush()
            continue

        if not in_body and _BODY_START.match(line):
            in_body = True
        if n in page_headers:
            header_page, header_line = page_headers[n], n
        elif _GAZETTE_PAGE.search(line):
            # Header with a misread number: still a new page (English pages run in steps of 2),
            # but only if the previous header was one page back; otherwise headers are missing.
            if header_page and n - header_line <= PAGE_MAX_LINES:
                header_page, header_line = header_page + 2, n
            else:
                header_page = None
        # Many headers are missing from the OCR; only trust a page header that is close above.
        page = header_page if header_page and n - header_line <= PAGE_MAX_LINES else None
        m_sched = _SCHEDULE.match(line) if in_body else None
        if m_sched:
            schedule = f"{m_sched.group(1).title()} Schedule"
            section, section_no, subsection_no = schedule, 0, 0
        elif in_body:
            m_sec = _SECTION_START.match(line)
            label = num_part = ""
            if m_sec:
                raw = m_sec.group(1)
                suffix = raw[-1] if len(raw) > 1 and raw[-1].isalpha() and raw[-1] not in "lIOS" else ""
                num_part = (raw[:-1] if suffix else raw).translate(_OCR_DIGITS)
                label = num_part + suffix
            num = int(num_part) if num_part.isdigit() else 0
            # OCR text has stray numbered lines; only accept the next few section numbers.
            if num and (section_no < num <= section_no + 5 or (num == section_no and suffix)):
                if num > section_no + 1 and section_no and not schedule:
                    gaps.append((section_start_line, n, section_no, num - 1))
                section_no = num
                section_start_line = n
                subsection_no = int(m_sec.group(2) or 0)
                base = f"{schedule} para {label}" if schedule else f"Section {label}"
                section = f"{base}({subsection_no})" if subsection_no else base
            elif section_no:
                m_sub = _SUBSECTION_START.match(line)
                # OCR turns "(b)" into "(5)" etc., so subsections must come in order.
                if m_sub and int(m_sub.group(1)) == subsection_no + 1:
                    subsection_no += 1
                    base = re.sub(r"\(\d+\)$", "", section)
                    section = f"{base}({subsection_no})"

        buf.append(Line(n, line, section, page))
    flush()

    # Where a heading was missed (e.g. 10 -> 12), the lines from the last heading up to the
    # next one could belong to either section, so label them with the range.
    for start, end, first, last in gaps:
        label = f"Sections {first}–{last} (heading not legible in scan)"
        for para in index.values():
            for line in para.lines:
                if start <= line.number < end:
                    line.section = label
    return index


def _locate(chunk: str) -> list[Unit]:
    """Single-section pieces of the Act that make up a retrieved chunk (overlap prefixes are skipped)."""
    index = _paragraph_index()
    found = []
    for para in re.split(r"\n{2,}", chunk):
        p = index.get(para.strip())
        if p:
            # The table of contents only lists titles; it is never a source for a point.
            found.extend(u for u in p.units()
                         if len(u.text) >= MIN_UNIT_CHARS and u.section != PREAMBLE)
    return found


@lru_cache(maxsize=1)
def section_titles() -> dict[str, str]:
    """Section number -> title, from the Act's table of contents ("20. Set-off of assessed losses")."""
    titles: dict[str, str] = {}
    current = None
    last = 0
    for line in LAW_TEXT_PATH.read_text(encoding="utf-8").split("\n"):
        if _BODY_START.match(line):
            break
        m = _SECTION_START.match(line)
        num = None
        if m:
            raw = m.group(1)
            suffix = raw[-1] if len(raw) > 1 and raw[-1].isalpha() and raw[-1] not in "lIOS" else ""
            digits = (raw[:-1] if suffix else raw).translate(_OCR_DIGITS)
            if digits.isdigit() and last < int(digits) <= last + 5:
                num = digits + suffix
        if num:
            last, current = int(re.match(r"\d+", num).group(0)), num
            titles[current] = line[m.end(1) + 1:]
        elif current and line.strip() and not re.match(r"^\s*(PART|CHAPTER|Section,)", line):
            titles[current] += " " + line  # title wrapped onto the next line
    # Strip OCR debris ("|", stray letters) from the ends of titles.
    return {k: re.sub(r"[\s|,;:._a]+$", "", re.sub(r"\s+", " ", v)).strip(" _") for k, v in titles.items()}


@lru_cache(maxsize=1)
def _title_vectors():
    titles = section_titles()
    keys = list(titles)
    return keys, _embedder()([f"Section {k}: {titles[k]}" for k in keys])


def sections_for_question(question: str, k: int = 3, min_similarity: float = 0.45) -> list[str]:
    """Section numbers whose titles best match the question, best first."""
    keys, vecs = _title_vectors()
    qv = _embedder()([question])[0]
    scored = sorted(((_cosine(qv, v), key) for key, v in zip(keys, vecs)), reverse=True)
    return [key for score, key in scored[:k] if score >= min_similarity]


def section_text(number: str) -> str:
    """Full text of one section, as whole paragraphs (so citations can locate them)."""
    paras = [p.text for p in _paragraph_index().values()
             if any(_unit_sections(Unit("", 0, 0, line.section, None)) & {number} for line in p.lines)]
    return "\n\n".join(paras)


def is_contents_chunk(chunk: str) -> bool:
    """True if a retrieved chunk is only the preamble / table of contents (titles, no provisions)."""
    index = _paragraph_index()
    paras = [index.get(p.strip()) for p in re.split(r"\n{2,}", chunk)]
    known = [p for p in paras if p]
    return bool(known) and all(l.section == PREAMBLE for p in known for l in p.lines)


@lru_cache(maxsize=1)
def _embedder():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def split_points(answer: str) -> list[str]:
    """Each paragraph or list item of the answer is one point."""
    points: list[str] = []
    for block in re.split(r"\n\s*\n", answer.strip()):
        current: list[str] = []
        for line in block.split("\n"):
            if _LIST_ITEM.match(line) and current:
                points.append("\n".join(current).strip())
                current = []
            current.append(line)
        if current:
            points.append("\n".join(current).strip())
    return [p for p in points if p]


def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


_NAMED_SECTION = re.compile(r"\bsections?\s+(\d{1,3}[A-Z]?)", re.I)
_UNIT_SECTION_NUMBERS = re.compile(r"^Sections?\s+(\d{1,3}[A-Z]?)(?:–(\d{1,3}))?")


def _unit_sections(unit: Unit) -> set[str]:
    """Section numbers a piece of the Act belongs to ("Sections 10–11 ..." gives both)."""
    m = _UNIT_SECTION_NUMBERS.match(unit.section)
    if not m:
        return set()
    if m.group(2):
        return {str(n) for n in range(int(m.group(1)), int(m.group(2)) + 1)}
    return {m.group(1).upper()}


def cite_answer(answer: str, chunks: list[str]) -> dict:
    """Split the answer into points and attach the best-matching piece of the Act to each.

    Returns {"points": [{"text", "source_id", "score", "kind"}], "sources": [...]}.
    kind is "cited", "unsupported" (nothing retrieved is close enough, or it names a section
    whose text doesn't support it), "summary" (the opening answer or closing summary around
    the bullets, which restates them and names no section) or "disclaimer".
    """
    points = split_points(answer)
    has_bullets = any(_LIST_ITEM.match(p) for p in points)
    candidates: list[Unit] = []
    seen = set()
    for chunk in chunks:
        for u in _locate(chunk):
            if u.line_start not in seen:
                seen.add(u.line_start)
                candidates.append(u)

    # Only a short point that is mainly a disclaimer skips checking; a long paragraph that
    # merely ends with "consult a tax practitioner" still makes claims that need a source.
    to_score = [i for i, t in enumerate(points)
                if not (_DISCLAIMER.search(t) and len(t) <= DISCLAIMER_MAX_CHARS)]
    vec_for = {}
    cand_vecs = []
    if to_score and candidates:
        embed = _embedder()
        vec_for = dict(zip(to_score, embed([points[i] for i in to_score])))
        cand_vecs = embed([c.text for c in candidates])

    sources: list[dict] = []
    source_ids: dict[int, int] = {}
    out = []
    for i, text in enumerate(points):
        if i not in to_score:
            out.append({"text": text, "source_id": None, "score": None, "kind": "disclaimer"})
            continue
        named = {n.upper() for n in _NAMED_SECTION.findall(text)}
        if has_bullets and not _LIST_ITEM.match(text) and not named:
            # Opening answer / closing summary: restates the bullets, so it gets no badge of its own.
            out.append({"text": text, "source_id": None, "score": None, "kind": "summary"})
            continue
        scores = [_cosine(vec_for[i], cv) for cv in cand_vecs] if i in vec_for else []
        allowed = range(len(scores))
        if named:
            # A point that names a section may only be cited to that section's text; if the
            # model named a section it wasn't shown, nothing qualifies and the point is hidden.
            allowed = [j for j in allowed if _unit_sections(candidates[j]) & named]
        best = max(allowed, key=scores.__getitem__) if allowed else None
        if best is None or scores[best] < MIN_SIMILARITY:
            out.append({
                "text": text,
                "source_id": None,
                "score": round(scores[best], 2) if best is not None else None,
                "kind": "unsupported",
            })
            continue
        unit = candidates[best]
        if unit.line_start not in source_ids:
            source_ids[unit.line_start] = len(sources) + 1
            excerpt = unit.text if len(unit.text) <= EXCERPT_CHARS else unit.text[:EXCERPT_CHARS].rstrip() + " ..."
            sources.append({
                "id": source_ids[unit.line_start],
                "section": unit.section,
                "line_start": unit.line_start,
                "line_end": unit.line_end,
                "page": unit.page,
                "excerpt": excerpt,
            })
        out.append({
            "text": text,
            "source_id": source_ids[unit.line_start],
            "score": round(scores[best], 2),
            "kind": "cited",
        })
    return {"points": out, "sources": sources}

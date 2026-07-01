import gc
import re
import textwrap
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import CHROMA_DIR, EMBEDDING_MODEL, LAW_TEXT_PATH, RAG_COLLECTION_NAME
from app.services.llm_service import generate

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def _clean_ocr_text(text: str) -> str:
    """Light cleanup of OCR artefacts without losing legal meaning."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split text into overlapping chunks, preserving paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[dict] = []
    current = ""
    chunk_id = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append({"id": f"chunk_{chunk_id}", "text": current.strip()})
            chunk_id += 1
            words = current.split()
            overlap_words = words[-overlap // 6:] if len(words) > overlap // 6 else []
            current = " ".join(overlap_words) + "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append({"id": f"chunk_{chunk_id}", "text": current.strip()})

    return chunks


def build_index(law_text_path: Path | None = None) -> None:
    """Read the law document, chunk it, embed it, and persist to ChromaDB."""
    path = law_text_path or LAW_TEXT_PATH
    print(f"Reading law text from {path} ...")
    raw = path.read_text(encoding="utf-8")
    cleaned = _clean_ocr_text(raw)
    chunks = _chunk_text(cleaned)
    print(f"Created {len(chunks)} chunks (size~{CHUNK_SIZE}, overlap~{CHUNK_OVERLAP})")

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if RAG_COLLECTION_NAME in existing:
        client.delete_collection(RAG_COLLECTION_NAME)
        print("Deleted old collection.")

    collection = client.get_or_create_collection(
        name=RAG_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
        )
        print(f"  Indexed chunks {i}..{i + len(batch) - 1}")

    print(f"Index built and persisted to {CHROMA_DIR.resolve()}")

    # Release references so cleanup runs before interpreter shutdown (avoids
    # Windows ResourceTracker / RLock exception on exit with sentence-transformers).
    del collection
    del client
    del embed_fn
    gc.collect()


def _get_collection():
    """Return the ChromaDB collection (must already be built)."""
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(
        name=RAG_COLLECTION_NAME,
        embedding_function=embed_fn,
    )


def retrieve(question: str, n_results: int = 8) -> list[str]:
    """Return the top-n most relevant chunks for a question."""
    collection = _get_collection()
    results = collection.query(query_texts=[question], n_results=n_results)
    return results["documents"][0]


SYSTEM_PROMPT = textwrap.dedent("""\
    You are a South African tax law expert assistant. You answer questions
    strictly based on the South African Income Tax Act No. 58 of 1962
    (the excerpts provided below).

    Rules:
    - Only use information from the CONTEXT sections provided.
    - If the context does not contain enough information, say so clearly.
    - Cite specific section numbers when possible.
    - Give practical, clear explanations suitable for a taxpayer.
    - Always add a disclaimer that this is informational only, not professional tax advice.
""")


def query(question: str, n_results: int = 8) -> str:
    """Full RAG pipeline: retrieve context, then generate an answer with the local LLM."""
    chunks = retrieve(question, n_results=n_results)

    context_block = "\n\n---\n\n".join(
        f"[CONTEXT {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    return generate(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context_block}\n\n"
                    f"QUESTION:\n{question}"
                ),
            },
        ],
        max_tokens=2048,
        temperature=0.2,
    )

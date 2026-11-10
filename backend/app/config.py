"""Central place for reading configuration and shared paths.

Loads .env once on import so every module sees the same values, regardless
of whether the caller exported them in the shell.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# .env wins over the shell, except in Docker (DOTENV_OVERRIDE=0), where the compose file's
# settings for the container must win over a .env written for running natively.
load_dotenv(override=os.environ.get("DOTENV_OVERRIDE", "1") != "0")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CHROMA_DIR = DATA_DIR / "chroma_db"
LAW_TEXT_PATH = DATA_DIR / "za-act-1962-58-publication-document.txt"
LAW_PDF_PATH = DATA_DIR / "za-act-1962-58-publication-document.pdf"
CORPUS_PATH = DATA_DIR / "tax_corpus.txt"
TRAINING_DATA_PATH = DATA_DIR / "tax_training_data.jsonl"
CAUSAL_LM_DIR = DATA_DIR / "tax_causal_lm"
LORA_ADAPTER_DIR = DATA_DIR / "tax_lora_adapter"
BANK_TRANSACTIONS_CSV = DATA_DIR / "bank_transactions_2025.csv"
CHAT_HISTORY_PATH = DATA_DIR / "chat_history.json"
USERS_DB_PATH = DATA_DIR / "users.db"
# One JSON file per chat, per user; lives in data/ so OneDrive syncs it across devices.
CONVERSATIONS_DIR = DATA_DIR / "conversations"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_COLLECTION_NAME = "sa_income_tax_act"
DEFAULT_HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_env(name: str, default: str = "") -> str:
    """Return an optional env var, falling back to default if unset/blank."""
    return (os.environ.get(name) or default).strip()


def require_env(name: str) -> str:
    """Return a required env var (e.g. an API key), or raise a clear error."""
    value = get_env(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to your .env file (e.g. {name}=...) "
            f"or export it as an environment variable."
        )
    return value


OPENAI_API_KEY = get_env("OPENAI_API_KEY")
OPENAI_MODEL = get_env("OPENAI_MODEL", "gpt-4.1-mini")
HF_MODEL = get_env("HF_MODEL", DEFAULT_HF_MODEL)
TAX_MODEL_ADAPTER = get_env("TAX_MODEL_ADAPTER")
LOCAL_LLM_DEVICE = get_env("LOCAL_LLM_DEVICE")
# Load the local model in 4-bit (needs bitsandbytes + CUDA); lets a 7B model fit in ~6 GB VRAM.
LOCAL_LLM_4BIT = get_env("LOCAL_LLM_4BIT", "0").lower() in ("1", "true", "yes")
# Max prompt + reply tokens for the local model (0 = model's own limit).
LOCAL_LLM_MAX_CONTEXT = int(get_env("LOCAL_LLM_MAX_CONTEXT", "8192"))
TAX_TRAIN_DEVICE = get_env("TAX_TRAIN_DEVICE")

# Login. JWT_SECRET signs session tokens; if unset, a random one is used per run
# (everyone is logged out whenever the backend restarts).
JWT_SECRET = get_env("JWT_SECRET")
JWT_EXPIRES_MINUTES = int(get_env("JWT_EXPIRES_MINUTES", "480"))
# Account created on first start, only while the users table is empty.
ADMIN_USERNAME = get_env("ADMIN_USERNAME", "Admin")
ADMIN_PASSWORD = get_env("ADMIN_PASSWORD", "Admin123")


def use_fine_tuned_model() -> bool:
    """True if a fine-tuned adapter is loaded (model was trained on the Act)."""
    return bool(TAX_MODEL_ADAPTER and os.path.isdir(TAX_MODEL_ADAPTER))

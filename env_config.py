"""Central place for reading configuration from the environment / .env file.

Loads .env once on import so every module sees the same values, regardless
of whether the caller exported them in the shell.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def require_env(name: str) -> str:
    """Return a required env var (e.g. an API key), or raise a clear error."""
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to your .env file (e.g. {name}=...) "
            f"or export it as an environment variable."
        )
    return value


def get_env(name: str, default: str = "") -> str:
    """Return an optional env var, falling back to default if unset/blank."""
    return (os.environ.get(name) or default).strip()

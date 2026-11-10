"""Lets a user stop a chat request that is still generating.

The chat route registers the request under a client-chosen id; a separate cancel request
sets its event, and generate() checks it between tokens and stops early.
"""
import threading
from contextlib import contextmanager
from contextvars import ContextVar

_lock = threading.Lock()
_events: dict[tuple[str, str], threading.Event] = {}
_current: ContextVar[threading.Event | None] = ContextVar("cancel_event", default=None)


class GenerationCancelled(Exception):
    """Raised when the user stopped the request."""


@contextmanager
def cancellable(username: str, request_id: str | None):
    """Make the enclosed work stoppable via cancel(username, request_id)."""
    event = threading.Event()
    key = (username.lower(), request_id) if request_id else None
    if key:
        with _lock:
            _events[key] = event
    token = _current.set(event)
    try:
        yield event
    finally:
        _current.reset(token)
        if key:
            with _lock:
                _events.pop(key, None)


def cancel(username: str, request_id: str) -> bool:
    """Stop a running request. Returns False if it isn't running (already finished, or unknown)."""
    with _lock:
        event = _events.get((username.lower(), request_id))
    if event is None:
        return False
    event.set()
    return True


def current_cancel_event() -> threading.Event | None:
    return _current.get()


def raise_if_cancelled() -> None:
    event = _current.get()
    if event is not None and event.is_set():
        raise GenerationCancelled()

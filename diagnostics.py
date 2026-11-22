"""
Shared diagnostics and recoverable-exception helpers for ComfyUI-OmniLoRA.

Provides:
- A dedicated `omni_lora` logger hierarchy that integrates cleanly with ComfyUI's logging.
- `log_once(...)`: thread-safe, memory-bounded deduplicated logging for hot-loop fallback paths
  so unexpected errors surface immediately without flooding the console.
- `strict_errors_enabled()`: honors `OMNILORA_STRICT_ERRORS=1` so CI/test runs can re-raise
  unexpected exceptions at critical fallback sites instead of silently falling back.
- `report_exception(...)`: standardized handler for non-fatal recovery, cleanup, and probing sites.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Set, Tuple

LOGGER_NAME = "omni_lora"
_MAX_SEEN_KEYS = 2048
_SEEN_LOCK = threading.Lock()
_SEEN_KEYS: Set[Tuple[str, str]] = set()


def get_logger(suffix: Optional[str] = None) -> logging.Logger:
    """Return the shared `omni_lora` logger (or a child logger when `suffix` is given)."""
    if suffix:
        return logging.getLogger(f"{LOGGER_NAME}.{suffix}")
    return logging.getLogger(LOGGER_NAME)


def strict_errors_enabled() -> bool:
    """Return True when `OMNILORA_STRICT_ERRORS` is enabled in the environment."""
    val = os.environ.get("OMNILORA_STRICT_ERRORS", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def clear_log_once_cache() -> None:
    """Reset deduplicated log keys (e.g. between training runs or unit tests)."""
    with _SEEN_LOCK:
        _SEEN_KEYS.clear()


def log_once(
    key: str,
    level: int,
    msg: str,
    exc: Optional[BaseException] = None,
    *,
    logger_name: Optional[str] = None,
) -> bool:
    """
    Log `msg` at `level` at most once per unique `(key, exc_type)` pair.

    Returns True if a log record was emitted, False if it was deduplicated.
    """
    exc_tag = type(exc).__name__ if exc is not None else ""
    dedup_key = (key, exc_tag)
    with _SEEN_LOCK:
        if dedup_key in _SEEN_KEYS:
            return False
        if len(_SEEN_KEYS) >= _MAX_SEEN_KEYS:
            _SEEN_KEYS.clear()
        _SEEN_KEYS.add(dedup_key)

    logger = get_logger(logger_name)
    if exc is not None:
        logger.log(level, "[OmniLoRA] %s (%s: %s)", msg, exc_tag, exc)
    else:
        logger.log(level, "[OmniLoRA] %s", msg)
    return True


def report_exception(
    key: str,
    msg: str,
    exc: BaseException,
    *,
    level: int = logging.DEBUG,
    once: bool = False,
    reraise_in_strict: bool = False,
    logger_name: Optional[str] = None,
) -> None:
    """
    Record a caught recoverable exception consistently.

    - When `reraise_in_strict=True` and `OMNILORA_STRICT_ERRORS=1`, re-raises `exc`.
    - When `once=True`, deduplicates per `(key, type(exc).__name__)` for hot-loop safety.
    - Otherwise emits a single structured log message at `level`.
    """
    if reraise_in_strict and strict_errors_enabled():
        raise exc
    if once:
        log_once(key, level, msg, exc=exc, logger_name=logger_name)
    else:
        get_logger(logger_name).log(
            level,
            "[OmniLoRA] %s (%s: %s)",
            msg,
            type(exc).__name__,
            exc,
        )

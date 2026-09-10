"""
This is the one place that knows a transient API failure from a real one.

Every model and embedding call in the Keeper goes through a `try/except` that degrades
rather than crashes, which is right: the water must never 500 at them. The cost was that
a RATE LIMIT looked exactly like a permanent failure, and the degradation was silent. A
fact embedded during a 429 was written to disk with embedding=None and never revisited,
so a transient error became permanent, keyword-only memory. Recall dropped to
keyword-only mid-turn, which is what made the recall-gate eval flake under load. The
supersede judge's verdict was lost, so a real change was filed DISTINCT. And a chat turn
fell back to the canned ERROR_LINE. All four are the same bug: no retry, and no word
about it. Retrying the handful of failures that are actually temporary removes most of
them, and logging the rest means the remaining degradation is visible instead of
invented.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

# Errors worth trying again. Matched by NAME so this module never has to import
# the openai SDK (it is used by modules that must work without a key at all).
_TRANSIENT_NAMES = frozenset({
    "RateLimitError", "APITimeoutError", "APIConnectionError",
    "InternalServerError", "ServiceUnavailableError", "Timeout",
    "TimeoutError", "ConnectionError",
})
_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0


def is_transient(exc: BaseException) -> bool:
    """True if `exc` is the kind of failure that may succeed on a second try."""
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status in _TRANSIENT_STATUS


def with_retry(fn: Callable[[], T], *, what: str,
               attempts: int = DEFAULT_ATTEMPTS,
               base_delay: float = DEFAULT_BASE_DELAY,
               sleep: Callable[[float], None] = time.sleep) -> T:
    """Call `fn`, retrying transient failures with exponential backoff.

    Permanent failures (a bad key, a malformed request) raise immediately: there
    is nothing to wait for. The final failure is re-raised either way, so callers
    keep whatever degradation they already had; they just get it less often, and
    the attempts are on the record.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            if not is_transient(exc) or attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[retry] {what}: {type(exc).__name__} on attempt "
                  f"{attempt}/{attempts}; waiting {delay:.1f}s", flush=True)
            sleep(delay)
    raise AssertionError("unreachable")   # pragma: no cover

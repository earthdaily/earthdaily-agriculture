"""Client-side rate limiting for bulk extraction.

A :class:`RateLimiter` is a thread-safe token bucket shared by the worker threads of
one bulk run, so ``max_workers`` threads together send at most ``rate`` requests per
second. Bulk methods get one through :meth:`BaseExtractor._paced`; single-entity
calls are never throttled.

Pacing is per HTTP attempt, retries included. ``_paced`` takes one token when a
worker starts an entity and binds the limiter to that worker thread with
:func:`bound`, leaving a one-shot credit for that token. The shared retry helper
(``api_utils.retry_with_backoff_no_retry_on_400``) calls :func:`take_turn` before
every attempt: the first spends the credit, every later one (retries, paging,
follow-up requests) takes a token. A 429 calls :meth:`RateLimiter.penalize`, which
holds every worker of the run for the server's ``Retry-After``. Extractors that do
not use the helper are still paced per entity by ``_paced``.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager


def validate_rate_limit(rate_limit):
    """Return ``rate_limit`` as a float, or ``None`` when it is ``None`` (no limit).

    Raises:
        ValueError: if it is not a positive number.
    """
    if rate_limit is None:
        return None
    if isinstance(rate_limit, bool):
        raise ValueError(f"rate_limit must be a positive number of requests per second, got {rate_limit!r}")
    try:
        value = float(rate_limit)
    except (TypeError, ValueError):
        raise ValueError(f"rate_limit must be a positive number of requests per second, got {rate_limit!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"rate_limit must be a positive number of requests per second, got {rate_limit!r}")
    return value


class RateLimiter:
    """Token bucket: at most ``rate`` acquisitions per second, ``burst`` at once.

    With the default ``burst=1`` calls are spaced evenly (``1 / rate`` seconds apart),
    the safest reading of an API quota. A larger ``burst`` lets that many calls go
    back to back after an idle period, still averaging ``rate`` per second.

    ``clock`` and ``sleep`` are injectable for tests.
    """

    def __init__(self, rate: float, burst: int = 1, clock=time.monotonic, sleep=time.sleep):
        self.rate = validate_rate_limit(rate)
        if self.rate is None:
            raise ValueError("RateLimiter needs a rate; use no limiter for 'no limit'")
        if int(burst) < 1:
            raise ValueError(f"burst must be at least 1, got {burst!r}")
        self.burst = int(burst)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = float(self.burst)
        self._updated = clock()
        # No token is handed out before this time (set by penalize after a 429).
        self._hold_until = float("-inf")
        #: Number of calls let through, and total seconds callers spent waiting.
        self.acquired = 0
        self.waited_seconds = 0.0
        #: Number of 429 responses reported through :meth:`penalize`.
        self.throttled = 0

    def acquire(self) -> float:
        """Block until a call may start; return the seconds this caller waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                if now < self._hold_until:
                    wait = self._hold_until - now
                else:
                    # No credit accrues during a hold: refill counts from its end.
                    start = max(self._updated, self._hold_until)
                    self._tokens = min(self.burst, self._tokens + max(0.0, now - start) * self.rate)
                    self._updated = now
                    # Tolerance: float refill can land a hair under 1 and ask for a ~1e-16 s wait.
                    if self._tokens >= 1 - 1e-9:
                        self._tokens = max(0.0, self._tokens - 1)
                        self.acquired += 1
                        self.waited_seconds += waited
                        return waited
                    wait = (1 - self._tokens) / self.rate
            # Sleep outside the lock so other threads can refill and check meanwhile.
            self._sleep(wait)
            waited += wait

    def penalize(self, seconds: float) -> None:
        """The server answered 429: hold every caller for ``seconds`` (its ``Retry-After``).

        Drops any spare burst and hands out no token before the hold ends. Holds from
        several workers throttled at once overlap rather than add up: the latest end wins.
        """
        with self._lock:
            now = self._clock()
            self._tokens = min(self._tokens, 0.0)
            self._hold_until = max(self._hold_until, now + max(0.0, float(seconds)))
            self.throttled += 1

    def __repr__(self) -> str:
        return f"RateLimiter(rate={self.rate:g}/s, burst={self.burst})"


# ---------------------------------------------------------------------------
# Binding a run's limiter to its worker threads
# ---------------------------------------------------------------------------

_local = threading.local()


@contextmanager
def bound(limiter: RateLimiter, credit: bool = True):
    """Bind ``limiter`` to the current thread for the duration of one worker call.

    ``credit=True`` records that the caller already took a token for this worker
    (``_paced`` does), so the first :func:`take_turn` spends that credit instead of
    taking a second token. Always cleared on exit, so nothing leaks to the next task
    run on a reused pool thread.
    """
    previous = (getattr(_local, "limiter", None), getattr(_local, "credit", False))
    _local.limiter, _local.credit = limiter, credit
    try:
        yield limiter
    finally:
        _local.limiter, _local.credit = previous


def current_limiter() -> RateLimiter | None:
    """The limiter bound to this thread by :func:`bound`, or ``None``."""
    return getattr(_local, "limiter", None)


def take_turn(limiter: RateLimiter | None = None) -> float:
    """Wait for one request's turn; return the seconds waited (0 with no limiter).

    ``limiter`` defaults to the one bound to this thread. A one-shot credit left by
    :func:`bound` for that limiter is spent first.
    """
    bound_limiter = current_limiter()
    limiter = limiter if limiter is not None else bound_limiter
    if limiter is None:
        return 0.0
    if limiter is bound_limiter and getattr(_local, "credit", False):
        _local.credit = False
        return 0.0
    return limiter.acquire()

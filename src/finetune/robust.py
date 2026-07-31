"""Fault tolerance for long teacher-generation runs.

Generating distillation targets for a few thousand prompts against a remote
teacher (NVIDIA NIM) is a long-running, partially-failing, rate-limited job. The
naive loop — ``[client.call(r) for r in records]`` — has three industrial defects:

  * one transient 429/503 at record 1,900 destroys 1,899 successful generations;
  * bursting through a token bucket gets the whole key throttled;
  * a mid-run Colab disconnect (routine on the free tier) loses everything.

This module supplies the three missing pieces — retry with exponential backoff
and jitter, a token-bucket rate limiter, and a resumable JSONL checkpoint. All
three take injectable clock/sleep functions so their behaviour is deterministic
and unit-testable without any real waiting.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

# HTTP statuses worth retrying: throttling and transient server-side faults.
# 4xx other than 429 are caller errors — retrying them just burns quota.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def status_of(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status extraction across SDK exception shapes."""
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    val = getattr(resp, "status_code", None)
    return val if isinstance(val, int) else None


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter, capped.

    Full jitter (``U(0, base·2^n)``) rather than fixed backoff: when many workers
    fail at the same instant, deterministic delays make them retry in lockstep and
    re-trigger the same throttle. Randomising spreads the retry storm.
    """

    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True

    def should_retry(self, attempt: int, exc: BaseException) -> bool:
        """``attempt`` is 1-based: the number of tries already made."""
        if attempt >= self.max_attempts:
            return False
        status = status_of(exc)
        if status is None:
            # No status: connection resets, timeouts, DNS blips — all transient.
            return True
        return status in RETRYABLE_STATUS

    def delay_for(self, attempt: int, rand: Callable[[], float] = random.random) -> float:
        """Delay in seconds before the ``attempt``-th retry (1-based)."""
        raw = min(self.base_delay * (2 ** max(0, attempt - 1)), self.max_delay)
        return raw * rand() if self.jitter else raw


def retry_call(
    fn: Callable[[], object],
    policy: Optional[RetryPolicy] = None,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> object:
    """Call ``fn`` with retries. Re-raises the last exception when giving up."""
    policy = policy or RetryPolicy()
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - policy decides what is fatal
            if not policy.should_retry(attempt, exc):
                raise
            delay = policy.delay_for(attempt, rand)
            if on_retry:
                on_retry(attempt, exc, delay)
            sleep(delay)


class RateLimiter:
    """Token bucket smoothing calls to ``rate`` per second with burst capacity.

    Keeps a fractional token balance rather than sleeping a fixed interval per
    call, so short idle gaps bank capacity and a burst is allowed to spend it —
    which is how provider quotas actually behave.
    """

    def __init__(
        self,
        rate: float,
        burst: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate <= 0:
            raise ValueError("rate must be > 0 requests/second")
        self.rate = float(rate)
        self.capacity = float(burst if burst is not None else max(1.0, rate))
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def acquire(self) -> float:
        """Block until a token is available. Returns the seconds actually waited."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0
        deficit = 1.0 - self._tokens
        wait = deficit / self.rate
        self._sleep(wait)
        # Charge the wait against the bucket without re-reading a possibly-fake clock.
        self._tokens = 0.0
        self._last = self._clock()
        return wait


class JsonlCheckpoint:
    """Append-only ``{index, output}`` log enabling resume of a generation run.

    Append-only and flushed per write: a process killed mid-run leaves every
    completed record intact, and a truncated final line (killed mid-flush) is
    skipped on load rather than aborting the resume.
    """

    def __init__(self, path: str):
        self.path = path

    def load(self) -> Dict[int, str]:
        """Return ``{index: output}`` for completed records; empty if no file."""
        if not self.path or not os.path.exists(self.path):
            return {}
        done: Dict[int, str] = {}
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done[int(row["index"])] = str(row.get("output", ""))
                except (ValueError, KeyError, TypeError):
                    continue  # torn/corrupt final line — ignore, do not crash
        return done

    def append(self, index: int, output: str) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"index": index, "output": output}, ensure_ascii=False))
            fh.write("\n")
            fh.flush()


def pending_indices(total: int, done: Dict[int, str]) -> List[int]:
    """Indices still needing generation, in order."""
    return [i for i in range(total) if i not in done]


def merge_checkpoint(total: int, done: Dict[int, str]) -> List[str]:
    """Materialise the full output list, empty string for any still-missing index."""
    return [done.get(i, "") for i in range(total)]

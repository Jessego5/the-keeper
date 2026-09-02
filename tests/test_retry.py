"""Tier 1 — the transient-failure retry policy (retry.py). No key, no network:
a fake sleep and hand-rolled exceptions, so these run instantly and pin the
DECISION (retry or not) rather than any timing.
"""
import pytest

import retry

pytestmark = pytest.mark.unit


class RateLimitError(Exception):
    """Shaped like the SDK's — matched by class name, not by import."""


class BadRequestError(Exception):
    pass


@pytest.fixture
def slept():
    out = []
    return out, (lambda d: out.append(d))


def test_rate_limit_is_transient():
    assert retry.is_transient(RateLimitError())


def test_a_bad_request_is_not_transient():
    assert not retry.is_transient(BadRequestError())


@pytest.mark.parametrize("status", [429, 500, 503, 408])
def test_transient_status_codes(status):
    exc = Exception()
    exc.status_code = status
    assert retry.is_transient(exc)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_status_codes(status):
    exc = Exception()
    exc.status_code = status
    assert not retry.is_transient(exc)


def test_succeeds_without_retrying(slept):
    calls, sleep = slept
    assert retry.with_retry(lambda: "ok", what="t", sleep=sleep) == "ok"
    assert calls == []


def test_retries_a_transient_failure_then_succeeds(slept):
    calls, sleep = slept
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("429")
        return "ok"

    assert retry.with_retry(flaky, what="t", sleep=sleep) == "ok"
    assert attempts["n"] == 3
    assert calls == [1.0, 2.0]          # exponential backoff


def test_a_permanent_failure_is_not_retried(slept):
    calls, sleep = slept
    attempts = {"n": 0}

    def broken():
        attempts["n"] += 1
        raise BadRequestError("bad key")

    with pytest.raises(BadRequestError):
        retry.with_retry(broken, what="t", sleep=sleep)
    assert attempts["n"] == 1, "a permanent error must fail fast"
    assert calls == []


def test_gives_up_and_re_raises_so_callers_keep_their_fallback(slept):
    """Retry must not swallow the final failure: every call site already degrades
    gracefully (keyword-only recall, a plain answer, ERROR_LINE), and those paths
    only run if the exception still arrives."""
    calls, sleep = slept
    with pytest.raises(RateLimitError):
        retry.with_retry(lambda: (_ for _ in ()).throw(RateLimitError("429")),
                         what="t", attempts=3, sleep=sleep)
    assert len(calls) == 2              # slept between the 3 attempts

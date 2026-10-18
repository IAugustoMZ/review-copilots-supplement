"""Shared HTTP plumbing for connectors: polite User-Agent, retry-with-backoff on
transient failures, and honoring `Retry-After` on 429s. Every call returns a
(response_or_none, rate_limit_events, error_detail) tuple rather than raising, so
callers can always populate a search_runs row with a real status.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

USER_AGENT = "review_pipeline/0.1 (GenAI-industrial-copilots literature review; mailto:{email})"
MAX_ATTEMPTS = 4
BASE_BACKOFF_SECONDS = 1.0
# A server's Retry-After is an arbitrary, untrusted value -- confirmed live
# that OpenAlex returns Retry-After: 4727 (79 minutes) once its daily usage
# budget is exhausted. Honoring that verbatim inside a synchronous retry loop
# would hang the whole search pipeline for over an hour on a single request.
# Past this cap, waiting is pointless within one process's run anyway --
# treat it as a hard failure instead (status/error_detail still get logged,
# same as any other rate-limit exhaustion).
MAX_RETRY_AFTER_SECONDS = 30.0


@dataclass(frozen=True)
class RetryPolicy:
    """Per-connector retry shape. The defaults suit Crossref/OpenAlex, which
    hand out generous per-mailto budgets and rarely 429 on this workload.
    """

    max_attempts: int = MAX_ATTEMPTS
    base_backoff_seconds: float = BASE_BACKOFF_SECONDS
    max_retry_after_seconds: float = MAX_RETRY_AFTER_SECONDS


# Semantic Scholar without an API key shares one anonymous pool with every other
# unauthenticated client, so a 429 there means "another tenant is using it right
# now", not "you are over your budget" -- it clears on its own in tens of
# seconds. The default 1/2/4s ladder gives up well inside that window and drops
# the connector from the run entirely (confirmed live 2026-08-29: CE1 returned
# `rate limited after 4 attempts`, 0 records). Wait longer and try more often
# instead; the key request has a multi-day manual turnaround, so this is the
# only lever available. Costs at most ~2min of wall clock on a fully contended
# pool, against losing a connector's whole contribution to recall.
ANONYMOUS_POOL_RETRY = RetryPolicy(max_attempts=6, base_backoff_seconds=4.0, max_retry_after_seconds=60.0)


@dataclass
class FetchResult:
    response: httpx.Response | None
    rate_limit_events: int
    error_detail: str | None


def build_client(contact_email: str, extra_headers: dict | None = None, timeout: float = 20.0) -> httpx.Client:
    headers = {"User-Agent": USER_AGENT.format(email=contact_email)}
    if extra_headers:
        headers.update(extra_headers)
    return httpx.Client(headers=headers, timeout=timeout)


def get_with_retry(
    client: httpx.Client,
    url: str,
    params: dict | None = None,
    policy: RetryPolicy | None = None,
) -> FetchResult:
    """GET with retry on transient network errors and 5xx; honors Retry-After on 429.

    Does NOT retry on other 4xx (those are real query/client errors, not transient).
    """
    policy = policy or RetryPolicy()
    rate_limit_events = 0
    last_error: str | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            response = client.get(url, params=params)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < policy.max_attempts:
                time.sleep(policy.base_backoff_seconds * (2 ** (attempt - 1)))
                continue
            return FetchResult(None, rate_limit_events, last_error)

        if response.status_code == 429:
            rate_limit_events += 1
            retry_after = response.headers.get("Retry-After")
            wait_seconds = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else policy.base_backoff_seconds * (2 ** (attempt - 1))
            )
            if wait_seconds > policy.max_retry_after_seconds:
                return FetchResult(
                    response,
                    rate_limit_events,
                    f"rate limited; server requested a {wait_seconds:.0f}s retry-after, "
                    f"exceeds the {policy.max_retry_after_seconds:.0f}s cap -- not waited",
                )
            if attempt < policy.max_attempts:
                time.sleep(wait_seconds)
                continue
            return FetchResult(response, rate_limit_events, f"rate limited after {attempt} attempts")

        if response.status_code >= 500:
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if attempt < policy.max_attempts:
                time.sleep(policy.base_backoff_seconds * (2 ** (attempt - 1)))
                continue
            return FetchResult(response, rate_limit_events, last_error)

        if response.status_code >= 400:
            return FetchResult(response, rate_limit_events, f"HTTP {response.status_code}: {response.text[:200]}")

        return FetchResult(response, rate_limit_events, None)

    return FetchResult(None, rate_limit_events, last_error)

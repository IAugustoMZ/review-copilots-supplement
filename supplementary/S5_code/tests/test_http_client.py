from __future__ import annotations

import httpx
import respx

from review_pipeline.src.search.connectors.http_client import build_client, get_with_retry

URL = "https://example.com/api"


@respx.mock
def test_retries_5xx_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("review_pipeline.src.search.connectors.http_client.time.sleep", lambda _: None)
    route = respx.get(URL)
    route.side_effect = [httpx.Response(503), httpx.Response(200, json={"ok": True})]

    client = build_client("test@example.com")
    result = get_with_retry(client, URL)

    assert result.response is not None
    assert result.response.status_code == 200
    assert result.error_detail is None


@respx.mock
def test_honors_retry_after_on_429(monkeypatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("review_pipeline.src.search.connectors.http_client.time.sleep", sleep_calls.append)
    route = respx.get(URL)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(200, json={"ok": True}),
    ]

    client = build_client("test@example.com")
    result = get_with_retry(client, URL)

    assert result.response is not None
    assert result.response.status_code == 200
    assert result.rate_limit_events == 1
    assert sleep_calls == [2.0]


@respx.mock
def test_retry_after_exceeding_cap_fails_fast_without_sleeping(monkeypatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("review_pipeline.src.search.connectors.http_client.time.sleep", sleep_calls.append)
    # A real value seen live from OpenAlex once its daily usage budget is
    # exhausted -- honoring it verbatim would sleep for 79 minutes.
    respx.get(URL).mock(return_value=httpx.Response(429, headers={"Retry-After": "4727"}))

    client = build_client("test@example.com")
    result = get_with_retry(client, URL)

    assert sleep_calls == []  # never slept -- would have hung for over an hour
    assert result.response is not None
    assert result.response.status_code == 429
    assert result.rate_limit_events == 1
    assert "exceeds" in result.error_detail


@respx.mock
def test_gives_up_after_max_attempts_on_persistent_5xx(monkeypatch) -> None:
    monkeypatch.setattr("review_pipeline.src.search.connectors.http_client.time.sleep", lambda _: None)
    respx.get(URL).mock(return_value=httpx.Response(500, text="boom"))

    client = build_client("test@example.com")
    result = get_with_retry(client, URL)

    assert result.response is not None
    assert result.response.status_code == 500
    assert result.error_detail is not None


@respx.mock
def test_non_retryable_4xx_returns_immediately(monkeypatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("review_pipeline.src.search.connectors.http_client.time.sleep", sleep_calls.append)
    respx.get(URL).mock(return_value=httpx.Response(400, text="bad request"))

    client = build_client("test@example.com")
    result = get_with_retry(client, URL)

    assert result.response.status_code == 400
    assert result.error_detail is not None
    assert sleep_calls == []  # no retry attempted for a genuine client error

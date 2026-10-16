from __future__ import annotations

import json
import http.client
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


CONTENT_FILTER_MARKERS = (
    "content_filter",
    "content filter",
    "content_policy",
    "moderation",
    "safety filter",
)


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = "",
                 retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.body = body
        self.retryable = retryable

    @property
    def is_content_filter(self) -> bool:
        text = f"{self} {self.body}".lower()
        return any(marker in text for marker in CONTENT_FILTER_MARKERS)


@dataclass(frozen=True)
class Endpoint:
    alias: str
    base_url: str
    api_key_env: str

    def key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise ProviderError(f"missing environment variable {self.api_key_env}")
        return value


@dataclass(frozen=True)
class ChatResult:
    content: str
    provider: str
    finish_reason: str | None
    reasoning_content: str
    usage: dict

    @property
    def complete(self) -> bool:
        return bool(self.content.strip())


def kimi_endpoints() -> tuple[Endpoint, Endpoint]:
    return (
        Endpoint("kimi-k2.7-code-moonshot", "https://api.moonshot.cn/v1", "KIMI_MOONSHOT_API_KEY"),
        Endpoint("kimi-k2.7-code-ksyun", "https://kspmas.ksyun.com/v1", "KIMI_KSYUN_API_KEY"),
    )


def _resolve_primary_and_fallback() -> tuple[Endpoint, Endpoint | None]:
    """Resolve the primary OpenAI-compatible endpoint from the env contract.

    When OPENAI_API_BASE is set, that endpoint is used verbatim (token from
    OPENAI_API_TOKEN, falling back to the Kimi key) and there is no content-filter
    fallback.  Otherwise the Kimi defaults apply: Moonshot primary with Ksyun as the
    content-filter fallback, selectable via KIMI_PRIMARY_PROVIDER.
    """
    openai_base = os.environ.get("OPENAI_API_BASE")
    if openai_base:
        token_env = "OPENAI_API_TOKEN" if os.environ.get("OPENAI_API_TOKEN") else "KIMI_MOONSHOT_API_KEY"
        return Endpoint("openai", openai_base.rstrip("/"), token_env), None
    primary_provider = os.environ.get("KIMI_PRIMARY_PROVIDER", "moonshot")
    if primary_provider not in {"moonshot", "ksyun"}:
        raise ValueError("KIMI_PRIMARY_PROVIDER must be moonshot or ksyun")
    moonshot, ksyun = kimi_endpoints()
    if primary_provider == "ksyun":
        return ksyun, moonshot
    return moonshot, ksyun


def _messages_url(base: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _resolve_anthropic_endpoint() -> Endpoint | None:
    base = os.environ.get("ANTHROPIC_API_BASE")
    token = os.environ.get("ANTHROPIC_API_TOKEN")
    if not (base and token):
        return None
    return Endpoint("anthropic", _messages_url(base), "ANTHROPIC_API_TOKEN")


def _anthropic_request(endpoint: Endpoint, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint.base_url.rstrip("/") + "/messages",
        data=data,
        headers={
            "x-api-key": endpoint.key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:4000]
        raise ProviderError(f"{endpoint.alias} HTTP {exc.code}", status=exc.code, body=body) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError,
            http.client.RemoteDisconnected, ssl.SSLError) as exc:
        raise ProviderError(f"{endpoint.alias} network error: {exc}", retryable=True) from exc


def completion_from_anthropic_payload(payload: dict, provider: str) -> ChatResult:
    content = "".join(
        block.get("text", "")
        for block in payload.get("content") or []
        if block.get("type") == "text"
    )
    return ChatResult(
        content=content,
        provider=provider,
        finish_reason=payload.get("stop_reason"),
        reasoning_content="",
        usage=payload.get("usage") or {},
    )


def _anthropic_chat_detailed(
    messages: list[dict],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> ChatResult:
    endpoint = _resolve_anthropic_endpoint()
    if endpoint is None:
        raise ProviderError("ANTHROPIC_API_BASE/ANTHROPIC_API_TOKEN not configured")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }
    last_error: ProviderError | None = None
    for attempt in range(retries + 1):
        try:
            result = _anthropic_request(endpoint, payload, timeout)
            return completion_from_anthropic_payload(result, endpoint.alias)
        except ProviderError as exc:
            last_error = exc
            if (not exc.retryable and exc.status not in {408, 429, 500, 502, 503, 504}) \
                    or attempt == retries:
                raise
            time.sleep(min(2**attempt, 4))
    raise last_error or ProviderError("unreachable")


def _request(endpoint: Endpoint, path: str, payload: dict | None, timeout: int) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint.base_url.rstrip("/") + path,
        data=data,
        headers={"Authorization": f"Bearer {endpoint.key()}", "Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:4000]
        raise ProviderError(f"{endpoint.alias} HTTP {exc.code}", status=exc.code, body=body) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError,
            http.client.RemoteDisconnected, ssl.SSLError) as exc:
        raise ProviderError(f"{endpoint.alias} network error: {exc}", retryable=True) from exc


def list_models(endpoint: Endpoint, timeout: int = 30) -> list[str]:
    payload = _request(endpoint, "/models", None, timeout)
    return [item.get("id", "") for item in payload.get("data", []) if item.get("id")]


def completion_from_payload(payload: dict, provider: str) -> ChatResult:
    choices = payload.get("choices") or []
    if not choices:
        raise ProviderError(f"{provider} response has no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    # vllm reasoning models put the thinking trace in `message.reasoning`;
    # OpenAI-compatible front-ends use `message.reasoning_content`.  Prefer the
    # standard key, fall back to the vllm key, and record whichever is present.
    reasoning = (message.get("reasoning_content")
                 or message.get("reasoning")
                 or message.get("reasoning_text")
                 or "")
    return ChatResult(
        content=message.get("content") or "",
        provider=provider,
        finish_reason=choice.get("finish_reason"),
        reasoning_content=reasoning,
        usage=payload.get("usage") or {},
    )


def chat_detailed(
    messages: list[dict],
    *,
    model: str = "kimi-k2.7-code",
    max_tokens: int = 2048,
    temperature: float = 1.0,
    timeout: int = 180,
    retries: int | None = None,
    endpoint: Endpoint | None = None,
) -> ChatResult:
    """An explicit OpenAI endpoint uses the supplied model, without env routing.

    Existing callers keep environment-based routing, including provider fallback.
    """
    if retries is None:
        retries = int(os.environ.get("KIMI_PROVIDER_RETRIES", "2"))
    if endpoint is None:
        model = os.environ.get("MODEL_NAME") or model
    if endpoint is None and model.startswith("claude-") and _resolve_anthropic_endpoint():
        return _anthropic_chat_detailed(
            messages, model=model, max_tokens=max_tokens, temperature=temperature,
            timeout=timeout, retries=retries,
        )
    primary, fallback = (endpoint, None) if endpoint is not None else _resolve_primary_and_fallback()
    request_messages = list(messages)
    if os.environ.get("KIMI_FINAL_ONLY") == "1":
        request_messages.insert(0, {
            "role": "system",
            "content": (
                "Answer directly and concisely. You must return a non-empty final answer in the "
                "assistant content field and follow the user's requested ANSWER: format. Do not "
                "include step-by-step reasoning."
            ),
        })
    payload = {
        "model": model,
        "messages": request_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    thinking_type = os.environ.get("KIMI_THINKING_TYPE")
    if thinking_type:
        if thinking_type not in {"enabled", "disabled"}:
            raise ValueError("KIMI_THINKING_TYPE must be enabled or disabled")
        payload["thinking"] = {"type": thinking_type}
    last_error: ProviderError | None = None
    for attempt in range(retries + 1):
        try:
            result = _request(primary, "/chat/completions", payload, timeout)
            return completion_from_payload(result, primary.alias)
        except ProviderError as exc:
            last_error = exc
            if exc.is_content_filter:
                if fallback is None:
                    raise
                result = _request(fallback, "/chat/completions", payload, timeout)
                return completion_from_payload(result, fallback.alias)
            if (not exc.retryable and exc.status not in {408, 429, 500, 502, 503, 504}) \
                    or attempt == retries:
                raise
            time.sleep(min(2**attempt, 4))
    raise last_error or ProviderError("unreachable")


def chat(
    messages: list[dict],
    *,
    model: str = "kimi-k2.7-code",
    max_tokens: int = 2048,
    temperature: float = 1.0,
    timeout: int = 180,
    retries: int | None = None,
) -> tuple[str, str]:
    result = chat_detailed(
        messages, model=model, max_tokens=max_tokens, temperature=temperature,
        timeout=timeout, retries=retries,
    )
    return result.content, result.provider

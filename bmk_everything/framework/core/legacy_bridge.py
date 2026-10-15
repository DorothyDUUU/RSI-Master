from __future__ import annotations

import os

import requests

from .provider import ChatResult, chat_detailed, completion_from_payload


def call_completion(prompt: str, *, base_url: str, model: str, api_key: str,
                    max_tokens: int, temperature: float, timeout: int) -> ChatResult:
    return call_messages(
        [{"role": "user", "content": prompt}], base_url=base_url, model=model,
        api_key=api_key, max_tokens=max_tokens, temperature=temperature,
        timeout=timeout,
    )


def call_messages(messages: list[dict[str, str]], *, base_url: str, model: str,
                  api_key: str, max_tokens: int, temperature: float,
                  timeout: int) -> ChatResult:
    if model == "kimi-k2.7-code" and os.environ.get("KIMI_MOONSHOT_API_KEY"):
        return chat_detailed(
            messages, model=model, max_tokens=max_tokens,
            temperature=1.0, timeout=timeout,
        )
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return completion_from_payload(response.json(), "openai-compatible")


def completion_fields(completion: ChatResult) -> dict:
    answer_present = bool(completion.content.strip())
    if answer_present:
        model_failure_reason = None
    elif completion.finish_reason == "length":
        model_failure_reason = "length_without_answer"
    elif completion.reasoning_content.strip():
        model_failure_reason = "reasoning_only"
    else:
        model_failure_reason = "empty_content"
    return {
        "response": completion.content,
        "reasoning_content": completion.reasoning_content,
        "finish_reason": completion.finish_reason,
        "usage": completion.usage,
        "provider": completion.provider,
        # Reaching this function means the provider request succeeded.  Empty,
        # truncated, or reasoning-only model output is a completed attempt that
        # scores zero; only request exceptions are infrastructure-incomplete.
        "request_complete": True,
        "complete": True,
        "answer_present": answer_present,
        "model_failure_reason": model_failure_reason,
        "incomplete_reason": None,
    }


def parse_sample_indices() -> list[int] | None:
    """Return the explicitly selected dataset indices from the runner environment."""
    raw = os.environ.get("SAMPLE_INDICES")
    if raw is None or not raw.strip():
        return None
    try:
        indices = [int(item) for item in raw.split(",")]
    except ValueError as exc:
        raise ValueError("SAMPLE_INDICES must be a comma-separated list of integers") from exc
    if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
        raise ValueError("SAMPLE_INDICES must contain unique non-negative integers")
    return sorted(indices)

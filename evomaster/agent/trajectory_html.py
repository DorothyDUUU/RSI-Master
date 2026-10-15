from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)


def render_trajectory_html(trajectory_jsonl_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Render an append-only trajectory JSONL file into a standalone HTML view."""
    jsonl_path = Path(trajectory_jsonl_path)
    html_path = Path(output_path) if output_path is not None else jsonl_path.with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)

    headers: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "header":
                    headers.append(payload)
                elif payload.get("type") == "step":
                    steps.append(payload)

    html_path.write_text(_build_html(jsonl_path, headers, steps), encoding="utf-8")
    return html_path


def _build_html(jsonl_path: Path, headers: list[dict[str, Any]], steps: list[dict[str, Any]]) -> str:
    header = headers[0] if headers else {}
    task_id = _escape(header.get("task_id") or "unknown")
    agents = sorted(
        {
            str((step.get("meta") or {}).get("agent_name") or "")
            for step in steps
            if isinstance(step.get("meta"), dict) and (step.get("meta") or {}).get("agent_name")
        }
        | {str(item.get("agent_name")) for item in headers if item.get("agent_name")}
    )
    agent_name = _escape(", ".join(agents) if agents else "unknown")
    tool_names = {
        (((tool.get("function") or {}).get("name")) if isinstance(tool, dict) else "")
        for item in headers
        for tool in (item.get("tools") or [])
    }
    tool_count = len({name for name in tool_names if name})
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_cards = "\n".join(_render_step(step, index + 1) for index, step in enumerate(steps))
    if not step_cards:
        step_cards = '<section class="empty">No trajectory steps have been recorded yet.</section>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RSI-Master Trajectory - {task_id}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #20242a;
      --muted: #657083;
      --accent: #1f6feb;
      --tool: #7a4d00;
      --ok: #1a7f37;
      --warn: #b54708;
      --code: #f1f4f8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(8px);
    }}
    .top {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 16px 20px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 9px;
      background: #fff;
      color: var(--muted);
      white-space: nowrap;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px 20px 40px;
    }}
    .step {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      margin: 0 0 14px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }}
    .step-head {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      background: #fbfcfe;
    }}
    .step-title {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      font-weight: 650;
    }}
    .status {{
      color: var(--ok);
      font-weight: 600;
    }}
    .usage {{
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .body {{
      padding: 12px 14px 14px;
      display: grid;
      gap: 10px;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
      background: #fff;
    }}
    details[open] summary {{
      border-bottom: 1px solid var(--line);
    }}
    summary {{
      cursor: pointer;
      padding: 9px 10px;
      font-weight: 600;
      background: #fbfcfe;
    }}
    pre {{
      margin: 0;
      padding: 11px 12px;
      overflow: auto;
      max-height: 520px;
      background: var(--code);
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .message {{
      border-left: 4px solid var(--line);
    }}
    .message.assistant {{ border-left-color: var(--accent); }}
    .message.tool {{ border-left-color: var(--tool); }}
    .message.user {{ border-left-color: #7c3aed; }}
    .message.system {{ border-left-color: #64748b; }}
    .tool-call {{
      border-left: 4px solid var(--warn);
    }}
    .empty {{
      padding: 24px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: #fff;
    }}
    @media (max-width: 760px) {{
      .top, main {{ padding-left: 12px; padding-right: 12px; }}
      .top {{ grid-template-columns: 1fr; }}
      .step-head {{ align-items: flex-start; flex-direction: column; }}
      .usage {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="top">
      <div>
        <h1>RSI-Master Trajectory</h1>
        <div class="meta">
          <span>Task: <strong>{task_id}</strong></span>
          <span>Agents: <strong>{agent_name}</strong></span>
          <span>Source: <code>{_escape(str(jsonl_path))}</code></span>
        </div>
      </div>
      <div class="meta">
        <span class="pill">{len(steps)} steps</span>
        <span class="pill">{tool_count} tools</span>
        <span class="pill">Rendered {generated_at}</span>
      </div>
    </div>
  </header>
  <main>
    {step_cards}
  </main>
</body>
</html>
"""


def _render_step(step: dict[str, Any], ordinal: int) -> str:
    meta = step.get("meta") if isinstance(step.get("meta"), dict) else {}
    agent = _escape(meta.get("agent_name") or "unknown")
    status = _escape(step.get("status") or "running")
    step_id = _escape(step.get("task_id") or f"step-{ordinal}")
    usage = _usage_text(step)

    assistant = step.get("assistant_message")
    tool_responses = step.get("tool_responses") or []
    messages = []
    for message in step.get("messages_delta") or []:
        role = str(message.get("role") or "")
        if role == "assistant" and isinstance(assistant, dict):
            continue
        if role == "tool" and tool_responses:
            continue
        messages.append(_render_message(message))
    assistant = step.get("assistant_message")
    if isinstance(assistant, dict):
        assistant_html = _render_assistant_details(assistant)
        if assistant_html:
            messages.append(assistant_html)
    for response in tool_responses:
        messages.append(_render_message(response))
    body = "\n".join(messages) or '<section class="empty">No messages recorded for this step.</section>'

    return f"""<article class="step">
  <div class="step-head">
    <div class="step-title">
      <span>Step {ordinal}</span>
      <span class="pill">{agent}</span>
      <span class="pill">{step_id}</span>
      <span class="status">{status}</span>
    </div>
    <div class="usage">{_escape(usage)}</div>
  </div>
  <div class="body">
    {body}
  </div>
</article>"""


def _render_assistant_details(message: dict[str, Any]) -> str:
    chunks: list[str] = []
    content = message.get("content")
    if content:
        chunks.append(_details("Assistant message", content, css_class="message assistant", open_by_default=True))
    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        fn = ((tool_call or {}).get("function") or {})
        label = f"Tool call: {fn.get('name') or 'unknown'}"
        chunks.append(_details(label, fn.get("arguments") or tool_call, css_class="tool-call", open_by_default=True))
    reasoning = message.get("reasoning_content")
    if reasoning:
        chunks.append(_details("Reasoning content", reasoning, css_class="message assistant", open_by_default=True))
    return "\n".join(chunks)


def _render_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "message")
    name = message.get("name")
    if role == "tool":
        label = "Tool result" if not name else f"Tool result: {name}"
    else:
        label = role if not name else f"{role}: {name}"
    content = message.get("content")
    return _details(
        label,
        content if content is not None else message,
        css_class=f"message {role}",
        open_by_default=role == "tool",
    )


def _details(label: str, value: Any, *, css_class: str = "", open_by_default: bool = False) -> str:
    open_attr = " open" if open_by_default else ""
    return (
        f'<details class="{_escape(css_class)}"{open_attr}>'
        f"<summary>{_escape(label)}</summary>"
        f"<pre>{_escape(_stringify(_sanitize(value)))}</pre>"
        "</details>"
    )


def _usage_text(step: dict[str, Any]) -> str:
    assistant = step.get("assistant_message") if isinstance(step.get("assistant_message"), dict) else {}
    meta = assistant.get("meta") if isinstance(assistant.get("meta"), dict) else {}
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    if not usage:
        return ""
    parts = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key in usage:
            parts.append(f"{key}={usage[key]}")
    model = meta.get("model")
    if model:
        parts.append(f"model={model}")
    return " | ".join(parts)


def _sanitize(value: Any, key: str | None = None) -> Any:
    if key and _is_sensitive_key(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, str(k)) for k, v in value.items() if k != "raw_response"}
    if isinstance(value, list):
        return [_sanitize(item, key) for item in value]
    if isinstance(value, str):
        return _mask_secret_like_strings(value)
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _mask_secret_like_strings(value: str) -> str:
    words = value.split()
    masked = []
    for word in words:
        if word.startswith("sk-") and len(word) > 10:
            masked.append(word[:6] + "..." + word[-4:])
        else:
            masked.append(word)
    return " ".join(masked) if len(words) > 1 else (masked[0] if masked else value)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)

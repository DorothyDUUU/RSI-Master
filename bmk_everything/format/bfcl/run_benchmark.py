#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from numbers import Number
from pathlib import Path
from typing import Any

import requests
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "framework"))
from core.subset import select_subset, write_subset_manifest

BENCHMARK_ID = "bfcl"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_SHA256 = "0943d4b5ac828672eb268143a762d1a17216e925e357b50269c3ba2d17be5358"
DEFAULT_DATASET_ROWS = 5251
DEFAULT_OPENAI_BASE = "https://api.moonshot.cn/v1"
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_WORKERS = 6
CLASS_TO_FILE = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "PostingAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "TwitterAPI": "posting_api.json",
}


def is_placeholder(value: object) -> bool:
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def resolve_general(config: dict, key: str, env: str, default=None):
    value = (config.get("general") or {}).get(key)
    if value is None or is_placeholder(value):
        return os.environ.get(env) or default
    return value


def resolve_custom(config: dict, key: str, default=None):
    return (config.get("custom") or {}).get(key, default)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_output_dir(config: dict) -> Path:
    template = resolve_general(config, "output_dir", "OUTPUT_DIR", "data/output/<BENCHMARK_ID>/<TRIAL_NAME>")
    template = str(template)
    trial_name = os.environ.get("TRIAL_NAME", "default")
    value = template.replace("<BENCHMARK_ID>", BENCHMARK_ID).replace("<TRIAL_NAME>", trial_name)
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolved_config(config: dict, model: str, split: str, output_dir: Path) -> dict:
    result = copy.deepcopy(config)
    general = result.setdefault("general", {})
    general.update(
        {
            "model": model,
            "split": split,
            "openai_api_base": resolve_general(config, "openai_api_base", "OPENAI_API_BASE", DEFAULT_OPENAI_BASE),
            "openai_api_token": "<OPENAI_API_TOKEN>",
            "anthropic_api_base": resolve_general(config, "anthropic_api_base", "ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"),
            "anthropic_api_token": "<ANTHROPIC_API_TOKEN>",
            "output_dir": str(output_dir),
        }
    )
    return result


def normalize_schema(node: Any) -> Any:
    if isinstance(node, dict):
        normalized = {}
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                normalized[key] = {"dict": "object", "tuple": "array"}.get(value, value)
            else:
                normalized[key] = normalize_schema(value)
        if "type" not in normalized and "properties" in normalized:
            normalized["type"] = "object"
        return normalized
    if isinstance(node, list):
        return [normalize_schema(item) for item in node]
    return node


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=None)
def load_multi_turn_docs_for_classes(docs_root: str, classes: tuple[str, ...]) -> tuple[dict, ...]:
    root = Path(docs_root)
    docs: list[dict] = []
    for cls in classes:
        file_name = CLASS_TO_FILE.get(cls)
        if not file_name:
            raise RuntimeError(f"unknown multi-turn class {cls!r}")
        docs.extend(load_jsonl(root / file_name))
    return tuple(docs)


def build_tool_docs(raw_docs: list[dict]) -> tuple[list[dict], dict[str, str], dict[str, dict]]:
    tools: list[dict] = []
    alias_map: dict[str, str] = {}
    docs_by_name: dict[str, dict] = {}
    for doc in raw_docs:
        name = str(doc["name"])
        safe = name.replace(".", "_")
        if safe in alias_map and alias_map[safe] != name:
            raise RuntimeError(f"tool name collision after sanitization: {safe!r}")
        alias_map[name] = name
        alias_map[safe] = name
        docs_by_name[name] = doc
        docs_by_name[safe] = doc
        params = normalize_schema(copy.deepcopy(doc.get("parameters") or {}))
        params.setdefault("type", "object")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": safe,
                    "description": doc.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return tools, alias_map, docs_by_name


def canonicalize_name(name: str, alias_map: dict[str, str]) -> str:
    if name in alias_map:
        return alias_map[name]
    if "." in name:
        suffix = name.rsplit(".", 1)[-1]
        if suffix in alias_map:
            return alias_map[suffix]
    return name


def ast_function_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        else:
            raise ValueError("unsupported function expression")
        return ".".join(reversed(parts))
    raise ValueError("unsupported function expression")


def safe_literal_value(node: ast.AST) -> Any:
    """Evaluate literals plus arithmetic expressions used in BFCL gold calls."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        pass
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = safe_literal_value(node.operand)
        if not isinstance(value, Number):
            raise ValueError("unary operator requires a number")
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = safe_literal_value(node.left)
        right = safe_literal_value(node.right)
        if not isinstance(left, Number) or not isinstance(right, Number):
            raise ValueError("arithmetic operands must be numbers")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
    raise ValueError(f"unsupported literal expression: {ast.dump(node)}")


def parse_call_expression_strict(text: str) -> dict:
    parsed = ast.parse(text, mode="eval")
    call = parsed.body
    if not isinstance(call, ast.Call):
        raise ValueError(f"not a function call: {text!r}")
    function = ast_function_name(call.func)
    positional = [safe_literal_value(arg) for arg in call.args]
    arguments = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("**kwargs are not supported")
        arguments[keyword.arg] = safe_literal_value(keyword.value)
    return {"function": function, "positional": positional, "arguments": arguments}


def parse_call_expression(text: str) -> dict:
    try:
        return parse_call_expression_strict(text)
    except (ValueError, SyntaxError, TypeError):
        return parse_call_expression_lenient(text)


def split_call_arguments(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    quote = None
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if quote != "'" or next_char in ",)]}" or not next_char.isalnum():
                    quote = None
            elif char == "," and depth == 0 and re.match(r"\s*[A-Za-z_]\w*\s*[:=]", text[index + 1:]):
                parts.append(text[start:index])
                start = index + 1
                quote = None
        else:
            if char in "'\"":
                quote = char
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                parts.append(text[start:index])
                start = index + 1
        index += 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def find_argument_separator(text: str) -> tuple[int, str] | None:
    depth = 0
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if quote != "'" or next_char in ",)]}" or not next_char.isalnum():
                    quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and char in "=:":
            return index, char
    return None


def parse_lenient_value(text: str) -> Any:
    value = text.strip()
    if not value:
        return ""
    try:
        return safe_literal_value(ast.parse(value, mode="eval").body)
    except (ValueError, SyntaxError, TypeError):
        pass
    if value[0] in "'\"":
        value = value[1:]
        if value and value[-1] in "'\"":
            value = value[:-1]
        return value.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
    return value


def parse_call_expression_lenient(text: str) -> dict:
    repaired = re.sub(
        r"([,(]\s*)'([A-Za-z_]\w*)='([^']*)'",
        r"\1\2='\3'",
        text.strip(),
    )
    match = re.match(r"\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", repaired)
    if not match:
        raise ValueError(f"not a function call: {text!r}")
    function = match.group(1)
    body = repaired[match.end():]
    if body.endswith(")"):
        body = body[:-1]
    positional = []
    arguments = {}
    for part in split_call_arguments(body):
        separator = find_argument_separator(part)
        if separator is None:
            positional.append(parse_lenient_value(part))
            continue
        index, _ = separator
        name = part[:index].strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValueError(f"invalid argument name: {name!r}")
        arguments[name] = parse_lenient_value(part[index + 1:])
    return {"function": function, "positional": positional, "arguments": arguments}


def canonicalize_parsed_call(parsed: dict, alias_map: dict[str, str], docs_by_name: dict[str, dict]) -> dict:
    name = canonicalize_name(str(parsed.get("function") or ""), alias_map)
    arguments = dict(parsed.get("arguments") or {})
    positional = list(parsed.get("positional") or [])
    if positional:
        doc = docs_by_name.get(name) or docs_by_name.get(canonicalize_name(name, alias_map))
        param_names = list(((doc or {}).get("parameters") or {}).get("properties") or {})
        for index, value in enumerate(positional):
            if index < len(param_names) and param_names[index] not in arguments:
                arguments[param_names[index]] = value
            else:
                arguments[f"__arg{index}"] = value
    return {"function": name, "arguments": arguments}


def parse_text_tool_calls(content: str, alias_map: dict[str, str], docs_by_name: dict[str, dict]) -> list[dict]:
    text = (content or "").strip()
    if not text:
        return []

    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if blocks:
        calls = []
        for block in blocks:
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                if "name" in payload and "arguments" in payload:
                    payload = {"function": payload.get("name"), "arguments": payload.get("arguments") or {}}
                elif "function" in payload and "arguments" in payload:
                    payload = {"function": payload.get("function"), "arguments": payload.get("arguments") or {}}
                if isinstance(payload.get("arguments"), str):
                    try:
                        payload["arguments"] = json.loads(payload["arguments"])
                    except json.JSONDecodeError:
                        payload["arguments"] = {}
                calls.append(canonicalize_parsed_call({"function": payload.get("function"), "arguments": payload.get("arguments") or {}}, alias_map, docs_by_name))
        if calls:
            return calls

    if text.startswith("[") and text.endswith("]"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            calls = []
            for item in payload:
                if isinstance(item, dict):
                    if "name" in item and "arguments" in item:
                        item = {"function": item.get("name"), "arguments": item.get("arguments") or {}}
                    elif "function" in item and "arguments" in item:
                        item = {"function": item.get("function"), "arguments": item.get("arguments") or {}}
                    if isinstance(item.get("arguments"), str):
                        try:
                            item["arguments"] = json.loads(item["arguments"])
                        except json.JSONDecodeError:
                            item["arguments"] = {}
                    calls.append(canonicalize_parsed_call({"function": item.get("function"), "arguments": item.get("arguments") or {}}, alias_map, docs_by_name))
            if calls:
                return calls

    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            if "name" in payload and "arguments" in payload:
                payload = {"function": payload.get("name"), "arguments": payload.get("arguments") or {}}
            elif "function" in payload and "arguments" in payload:
                payload = {"function": payload.get("function"), "arguments": payload.get("arguments") or {}}
            if isinstance(payload.get("arguments"), str):
                try:
                    payload["arguments"] = json.loads(payload["arguments"])
                except json.JSONDecodeError:
                    payload["arguments"] = {}
            return [canonicalize_parsed_call({"function": payload.get("function"), "arguments": payload.get("arguments") or {}}, alias_map, docs_by_name)]

    calls = []
    for match in re.finditer(r"([A-Za-z0-9_.]+\([^()]*\))", text):
        try:
            parsed = parse_call_expression(match.group(1))
        except Exception:
            continue
        calls.append(canonicalize_parsed_call(parsed, alias_map, docs_by_name))
    return calls


def standardize_string(input_string: str) -> str:
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", input_string).lower().replace("'", '"')


def scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Number) and isinstance(right, Number):
        return float(left) == float(right)
    if isinstance(left, str) and isinstance(right, str):
        return standardize_string(left) == standardize_string(right)
    return left == right


def exact_match_value(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(exact_match_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(exact_match_value(a, b) for a, b in zip(left, right))
    return scalar_equal(left, right)


def allowed_match_item(value: Any, option: Any) -> bool:
    if option == "":
        return value in ("", None, [])
    if isinstance(option, dict):
        return isinstance(value, dict) and allowed_match_dict(value, option)
    if isinstance(option, (list, tuple)):
        if not isinstance(value, (list, tuple)):
            return False
        value_list = list(value)
        option_list = list(option)
        if len(value_list) != len(option_list):
            return False
        return all(allowed_match_item(item, spec) for item, spec in zip(value_list, option_list))
    return scalar_equal(value, option)


def allowed_match_value(value: Any, spec: Any) -> bool:
    if isinstance(spec, list):
        return any(allowed_match_item(value, option) for option in spec)
    return allowed_match_item(value, spec)


def allowed_match_dict(value: dict, spec: dict) -> bool:
    for key, allowed in spec.items():
        if key not in value:
            if isinstance(allowed, list) and "" in allowed:
                continue
            return False
        if not allowed_match_value(value[key], allowed):
            return False
    for key in value:
        if key not in spec:
            return False
    return True


def schema_accepts(value: Any, schema: dict) -> bool:
    schema_type = schema.get("type")
    if schema_type in (None, "object", "dict"):
        if not isinstance(value, dict):
            return False
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        for key in required:
            if key not in value:
                return False
        for key, item in value.items():
            if key not in props:
                return False
            if not schema_accepts(item, props[key]):
                return False
        return True
    if schema_type in ("array", "tuple"):
        if not isinstance(value, (list, tuple)):
            return False
        items = schema.get("items")
        if items:
            return all(schema_accepts(item, items) for item in value)
        return True
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type in ("float", "number"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    return True


def compare_call_lists_ordered(predicted: list[dict], expected: list[dict], comparator) -> bool:
    if len(predicted) != len(expected):
        return False
    return all(comparator(p, e) for p, e in zip(predicted, expected))


def compare_call_lists_unordered(predicted: list[dict], expected: list[dict], comparator) -> bool:
    if len(predicted) != len(expected):
        return False
    remaining = list(expected)
    for predicted_call in predicted:
        match_index = None
        for index, expected_call in enumerate(remaining):
            if comparator(predicted_call, expected_call):
                match_index = index
                break
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def parse_expected_call_string(call_text: str, alias_map: dict[str, str], docs_by_name: dict[str, dict]) -> dict:
    return canonicalize_parsed_call(parse_call_expression(call_text), alias_map, docs_by_name)


def compare_allowed_call(predicted: dict, expected_spec: dict, alias_map: dict[str, str]) -> bool:
    if len(expected_spec) != 1:
        return False
    expected_name, expected_args = next(iter(expected_spec.items()))
    if canonicalize_name(predicted["function"], alias_map) != canonicalize_name(expected_name, alias_map):
        return False
    return allowed_match_dict(predicted.get("arguments") or {}, expected_args)


def compare_exact_call(predicted: dict, expected: dict, alias_map: dict[str, str]) -> bool:
    if canonicalize_name(predicted["function"], alias_map) != canonicalize_name(expected["function"], alias_map):
        return False
    return exact_match_value(predicted.get("arguments") or {}, expected.get("arguments") or {})


def normalize_question(question: Any) -> list[list[dict]]:
    if isinstance(question, str):
        return [[{"role": "user", "content": question}]]
    if isinstance(question, list) and question and isinstance(question[0], dict):
        return [question]
    return question or []


def tool_docs_for_row(row: dict, docs_root: Path) -> list[dict]:
    if row.get("score_kind") == "multi_turn":
        classes = tuple(row.get("involved_classes") or [])
        return list(load_multi_turn_docs_for_classes(str(docs_root), classes))
    docs = row.get("function") or []
    if isinstance(docs, str):
        return []
    if isinstance(docs, dict):
        return [docs]
    return list(docs)


def build_tool_context(row: dict, docs_root: Path):
    raw_docs = tool_docs_for_row(row, docs_root)
    return build_tool_docs(raw_docs)


def is_context_overflow_response(response: requests.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message", "") if isinstance(error, dict) else ""
    normalized = str(message).lower()
    return response.status_code == 400 and (
        "maximum context length" in normalized
        or "max_tokens" in normalized
        or "max tokens" in normalized
    )


def call_model(
    messages: list[dict],
    *,
    base_url: str,
    model: str,
    token: str,
    tools: list[dict] | None,
    max_tokens: int,
    fallback_max_tokens: list[int] | None,
    temperature: float,
    timeout: int,
    retries: int,
) -> dict:
    budgets = [max_tokens]
    for value in fallback_max_tokens or []:
        budget = int(value)
        if 0 < budget < max_tokens and budget not in budgets:
            budgets.append(budget)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{base_url.rstrip('/')}/chat/completions"
    last_error = None
    for budget in budgets:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": budget,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        context_overflow = False
        for attempt in range(retries + 1):
            response = None
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                result["_bmk_requested_max_tokens"] = budget
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if response is not None and is_context_overflow_response(response):
                    context_overflow = True
                    break
                if attempt < retries:
                    time.sleep(2 ** attempt)
        if not context_overflow:
            raise last_error
    raise last_error


def extract_calls_from_response(response: dict, alias_map: dict[str, str], docs_by_name: dict[str, dict]) -> tuple[list[dict], str, str | None, dict | None]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    raw_content = message.get("content") or ""
    tool_calls = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") or {}
        name = canonicalize_name(str(function.get("name") or ""), alias_map)
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        tool_calls.append({"function": name, "arguments": arguments})
    if tool_calls:
        return tool_calls, raw_content, choice.get("finish_reason"), response.get("usage")
    return parse_text_tool_calls(raw_content, alias_map, docs_by_name), raw_content, choice.get("finish_reason"), response.get("usage")


def is_truncated(finish_reason: str | None, usage: dict | None, max_tokens: int) -> bool:
    if finish_reason == "length":
        return True
    if not usage:
        return False
    completion_tokens = usage.get("completion_tokens")
    return isinstance(completion_tokens, int) and completion_tokens >= max_tokens


def score_with_truncation(correct: bool | None, truncated: bool) -> bool | None:
    if truncated and correct is not None:
        return False
    return correct


def score_single_turn(row: dict, predicted_calls: list[dict], alias_map: dict[str, str], docs_by_name: dict[str, dict]) -> tuple[bool | None, list[dict]]:
    kind = row.get("score_kind")
    if kind == "ast":
        expected_specs = row.get("possible_answer") or []
        if row.get("category") in {"parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"}:
            comparator = lambda predicted, expected: compare_allowed_call(predicted, expected, alias_map)
            return compare_call_lists_unordered(predicted_calls, expected_specs, comparator), expected_specs
        if len(predicted_calls) != 1:
            return False, expected_specs
        ok = any(compare_allowed_call(predicted_calls[0], expected_spec, alias_map) for expected_spec in expected_specs)
        return ok, expected_specs
    if kind == "exec":
        expected_calls = [parse_expected_call_string(call, alias_map, docs_by_name) for call in (row.get("ground_truth") or []) if str(call).strip()]
        if row.get("category") in {"exec_parallel", "exec_parallel_multiple"}:
            comparator = lambda predicted, expected: compare_exact_call(predicted, expected, alias_map)
            return compare_call_lists_unordered(predicted_calls, expected_calls, comparator), expected_calls
        if len(predicted_calls) != 1:
            return False, expected_calls
        ok = any(compare_exact_call(predicted_calls[0], expected_call, alias_map) for expected_call in expected_calls)
        return ok, expected_calls
    if kind == "no_call":
        return len(predicted_calls) == 0, []
    if kind == "relevance":
        if not predicted_calls:
            return False, []
        for call in predicted_calls:
            name = canonicalize_name(call["function"], alias_map)
            doc = docs_by_name.get(name)
            if doc is None or not schema_accepts(call.get("arguments") or {}, doc.get("parameters") or {}):
                return False, []
        return True, []
    if kind == "unscored_rest":
        return None, []
    raise RuntimeError(f"unknown score kind {kind!r}")


def evaluate_multi_turn(row: dict, *, base_url: str, model: str, token: str, tools: list[dict], alias_map: dict[str, str], docs_by_name: dict[str, dict], max_tokens: int, fallback_max_tokens: list[int] | None, temperature: float, timeout: int, retries: int) -> dict:
    expected_turns = row.get("possible_answer") or []
    turns = normalize_question(row.get("question"))
    history: list[dict] = []
    turn_results = []
    overall_correct = True
    completed = True
    error = None
    for turn_index, turn_messages in enumerate(turns):
        expected_calls_raw = expected_turns[turn_index] if turn_index < len(expected_turns) else []
        expected_calls = [parse_expected_call_string(call, alias_map, docs_by_name) for call in expected_calls_raw if str(call).strip()]
        if not turn_messages:
            predicted_calls = []
            raw_content = ""
            finish_reason = None
            usage = None
            truncated = False
            requested_max_tokens = None
        else:
            try:
                response = call_model(history + turn_messages, base_url=base_url, model=model, token=token, tools=tools, max_tokens=max_tokens, fallback_max_tokens=fallback_max_tokens, temperature=temperature, timeout=timeout, retries=retries)
                predicted_calls, raw_content, finish_reason, usage = extract_calls_from_response(response, alias_map, docs_by_name)
                requested_max_tokens = int(response.get("_bmk_requested_max_tokens", max_tokens))
                truncated = is_truncated(finish_reason, usage, requested_max_tokens)
            except Exception as exc:  # noqa: BLE001
                completed = False
                overall_correct = False
                error = f"{type(exc).__name__}: {exc}"
                turn_results.append({
                    "turn_index": turn_index,
                    "expected_calls": expected_calls_raw,
                    "predicted_calls": [],
                    "raw_content": "",
                    "finish_reason": None,
                    "usage": None,
                    "requested_max_tokens": None,
                    "truncated": False,
                    "correct": False,
                    "error": error,
                })
                break
        turn_ok = score_with_truncation(
            compare_call_lists_ordered(
                predicted_calls,
                expected_calls,
                lambda predicted, expected: compare_exact_call(
                    predicted, expected, alias_map
                ),
            ),
            truncated,
        )
        overall_correct = overall_correct and turn_ok
        turn_results.append({
            "turn_index": turn_index,
            "expected_calls": expected_calls_raw,
            "predicted_calls": predicted_calls,
            "raw_content": raw_content,
            "finish_reason": finish_reason,
            "usage": usage,
            "requested_max_tokens": requested_max_tokens,
            "truncated": truncated,
            "correct": turn_ok,
            "error": None,
        })
        history.extend(turn_messages)
        history.append({"role": "assistant", "content": json.dumps(predicted_calls, ensure_ascii=False) if predicted_calls else (raw_content or "")})
    return {
        "correct": overall_correct if completed else False,
        "completed": completed,
        "error": error,
        "turn_results": turn_results,
    }


def evaluate_row(row: dict, *, base_url: str, model: str, token: str, max_tokens: int, fallback_max_tokens: list[int] | None, temperature: float, timeout: int, retries: int, docs_root: Path) -> dict:
    started = time.perf_counter()
    tools, alias_map, docs_by_name = build_tool_context(row, docs_root)
    tool_names = [item["function"]["name"] for item in tools]
    try:
        if row.get("score_kind") == "multi_turn":
            multi = evaluate_multi_turn(
                row,
                base_url=base_url,
                model=model,
                token=token,
                tools=tools,
                alias_map=alias_map,
                docs_by_name=docs_by_name,
                max_tokens=max_tokens,
                fallback_max_tokens=fallback_max_tokens,
                temperature=temperature,
                timeout=timeout,
                retries=retries,
            )
            return {
                "id": row.get("id"),
                "category": row.get("category"),
                "score_kind": row.get("score_kind"),
                "scoreable": bool(row.get("scoreable", True)),
                "question": row.get("question"),
                "path": row.get("path"),
                "involved_classes": row.get("involved_classes"),
                "tool_names": tool_names,
                "predicted": [turn.get("predicted_calls") or [] for turn in multi["turn_results"]],
                "expected": row.get("possible_answer") or [],
                "turn_results": multi["turn_results"],
                "correct": multi["correct"],
                "completed": multi["completed"],
                "error": multi["error"],
                "finish_reason": None,
                "usage": None,
                "truncated": any(bool(turn.get("truncated")) for turn in multi["turn_results"]),
                "latency_s": round(time.perf_counter() - started, 6),
                "raw_content": "",
            }

        request_messages = normalize_question(row.get("question"))
        if len(request_messages) != 1:
            raise RuntimeError("single-turn row has invalid question shape")
        response = call_model(request_messages[0], base_url=base_url, model=model, token=token, tools=tools if tools else None, max_tokens=max_tokens, fallback_max_tokens=fallback_max_tokens, temperature=temperature, timeout=timeout, retries=retries)
        predicted_calls, raw_content, finish_reason, usage = extract_calls_from_response(response, alias_map, docs_by_name)
        requested_max_tokens = int(response.get("_bmk_requested_max_tokens", max_tokens))
        truncated = is_truncated(finish_reason, usage, requested_max_tokens)
        correct, expected = score_single_turn(row, predicted_calls, alias_map, docs_by_name)
        correct = score_with_truncation(correct, truncated)
        return {
            "id": row.get("id"),
            "category": row.get("category"),
            "score_kind": row.get("score_kind"),
            "scoreable": bool(row.get("scoreable", True)),
            "question": row.get("question"),
            "function": row.get("function"),
            "path": row.get("path"),
            "involved_classes": row.get("involved_classes"),
            "tool_names": tool_names,
            "predicted": predicted_calls,
            "expected": expected,
            "correct": correct,
            "completed": True,
            "error": None,
            "finish_reason": finish_reason,
            "usage": usage,
            "requested_max_tokens": requested_max_tokens,
            "truncated": truncated,
            "latency_s": round(time.perf_counter() - started, 6),
            "raw_content": raw_content,
        }
    except Exception as exc:  # noqa: BLE001
        expected = row.get("possible_answer") or row.get("ground_truth") or []
        return {
            "id": row.get("id"),
            "category": row.get("category"),
            "score_kind": row.get("score_kind"),
            "scoreable": bool(row.get("scoreable", True)),
            "question": row.get("question"),
            "function": row.get("function"),
            "path": row.get("path"),
            "involved_classes": row.get("involved_classes"),
            "tool_names": tool_names,
            "predicted": [],
            "expected": expected,
            "correct": None,
            "completed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "finish_reason": None,
            "usage": None,
            "requested_max_tokens": None,
            "truncated": False,
            "latency_s": round(time.perf_counter() - started, 6),
            "raw_content": "",
        }


def load_rows(dataset_path: Path, expected_sha: str, limit: int | None) -> list[dict]:
    data_file = dataset_path / "default.jsonl"
    data = data_file.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != expected_sha:
        raise RuntimeError(f"dataset hash mismatch: expected {expected_sha}, got {sha}")
    rows = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        for key in ("id", "category", "question", "score_kind", "scoreable"):
            if key not in raw:
                raise RuntimeError(f"line {line_number}: missing required field {key!r}")
        if raw["score_kind"] in {"ast", "exec"} and "function" not in raw:
            raise RuntimeError(f"line {line_number}: missing function field for score kind {raw['score_kind']!r}")
        if raw["score_kind"] == "ast" and "possible_answer" not in raw:
            raise RuntimeError(f"line {line_number}: missing possible_answer for ast row")
        if raw["score_kind"] == "exec" and "ground_truth" not in raw:
            raise RuntimeError(f"line {line_number}: missing ground_truth for exec row")
        if raw["score_kind"] == "multi_turn" and ("possible_answer" not in raw or "path" not in raw or "involved_classes" not in raw):
            raise RuntimeError(f"line {line_number}: incomplete multi-turn row")
        rows.append(raw)
        if limit is not None and limit > 0 and len(rows) >= limit:
            break
    return rows


def dry_run(config: dict) -> int:
    dataset_path = resolve_repo_path(str(resolve_custom(config, "dataset_path", "data/datasets/bfcl")))
    expected_sha = str(resolve_custom(config, "dataset_sha256", DEFAULT_DATASET_SHA256))
    rows = load_rows(dataset_path, expected_sha, limit=1)
    scoreable = sum(1 for row in load_rows(dataset_path, expected_sha, limit=None) if row.get("scoreable", True))
    print(
        json.dumps(
            {
                "benchmark-id": BENCHMARK_ID,
                "status": "ok",
                "dataset": {
                    "path": str(dataset_path),
                    "rows": DEFAULT_DATASET_ROWS,
                    "validated_rows": len(rows),
                    "scoreable_rows": scoreable,
                    "sha256": expected_sha,
                },
                "adapter": "legacy_runner",
                "scorer": "bfcl_call_sequence_match",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def summarize_rows(results: list[dict]) -> dict:
    by_category = defaultdict(lambda: {"total": 0, "scoreable": 0, "correct": 0, "completed": 0, "incomplete": 0, "truncated": 0})
    scoreable_total = 0
    scoreable_correct = 0
    scoreable_completed = 0
    unscored_total = 0
    truncated_total = 0
    for row in results:
        category = row.get("category") or "unknown"
        bucket = by_category[category]
        bucket["total"] += 1
        if row.get("scoreable", True):
            bucket["scoreable"] += 1
            scoreable_total += 1
            if row.get("correct") is True:
                bucket["correct"] += 1
                scoreable_correct += 1
            if row.get("completed"):
                bucket["completed"] += 1
                scoreable_completed += 1
            else:
                bucket["incomplete"] += 1
        else:
            unscored_total += 1
        if row.get("truncated"):
            bucket["truncated"] += 1
            truncated_total += 1
    category_stats = {
        category: {
            **stats,
            "accuracy": (stats["correct"] / stats["scoreable"] if stats["scoreable"] else None),
        }
        for category, stats in sorted(by_category.items())
    }
    accuracy = scoreable_correct / scoreable_total if scoreable_total else 0.0
    return {
        "scoreable_total": scoreable_total,
        "scoreable_correct": scoreable_correct,
        "scoreable_completed": scoreable_completed,
        "unscored_total": unscored_total,
        "accuracy": accuracy,
        "truncated_total": truncated_total,
        "category_stats": category_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BFCL v3 through BmkEverything")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--resume-from", help="reuse completed rows from an earlier samples.jsonl")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    split = args.split or resolve_general(config, "split", "SPLIT_NAME", "default")
    if split != "default":
        raise SystemExit(f"error: unsupported split {split!r}; available: ['default']")
    if args.dry_run or args.validate_data:
        raise SystemExit(dry_run(config))

    started = time.time()
    model = str(resolve_general(config, "model", "MODEL_NAME", "kimi-k2.7-code"))
    base_url = str(resolve_general(config, "openai_api_base", "OPENAI_API_BASE", DEFAULT_OPENAI_BASE))
    token = str(resolve_general(config, "openai_api_token", "OPENAI_API_TOKEN", ""))
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_repo_path(str(resolve_custom(config, "dataset_path", "data/datasets/bfcl")))
    docs_root = resolve_repo_path(str(resolve_custom(config, "multi_turn_docs_path", "data/datasets/bfcl/multi_turn_func_doc")))
    dataset_sha = str(resolve_custom(config, "dataset_sha256", DEFAULT_DATASET_SHA256))
    all_rows = load_rows(dataset_path, dataset_sha, limit=None)
    selection = select_subset(
        all_rows,
        limit=args.limit,
        fingerprint=dataset_sha,
        stable_key=lambda row: row["id"],
        stratum=lambda row: row["category"],
    )
    rows = [row for _, row in selection.rows]
    write_subset_manifest(output_dir, selection.manifest)
    preserved = {}
    if args.resume_from:
        previous_rows = load_jsonl(Path(args.resume_from))
        if len(previous_rows) != len(rows):
            raise RuntimeError("resume samples and dataset have different row counts")
        for index, (row, previous) in enumerate(zip(rows, previous_rows)):
            if row.get("id") != previous.get("id"):
                raise RuntimeError(f"resume row mismatch at index {index}")
            if previous.get("completed") and not previous.get("truncated"):
                preserved[index] = previous
    rows_to_evaluate = [row for index, row in enumerate(rows) if index not in preserved]
    max_workers = int(resolve_custom(config, "max_workers", DEFAULT_MAX_WORKERS))
    max_tokens = int(resolve_custom(config, "max_tokens", DEFAULT_MAX_TOKENS))
    fallback_max_tokens = resolve_custom(config, "fallback_max_tokens", [])
    if not isinstance(fallback_max_tokens, list):
        raise RuntimeError("custom.fallback_max_tokens must be a list")
    temperature = float(resolve_custom(config, "temperature", 0.0))
    timeout = int(resolve_custom(config, "timeout", DEFAULT_TIMEOUT))
    retries = int(resolve_custom(config, "retries", 2))

    print(f"[bfcl] model={model} split={split} samples={len(rows)} to_evaluate={len(rows_to_evaluate)} preserved={len(preserved)}")
    results_by_index = dict(preserved)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_row,
                row,
                base_url=base_url,
                model=model,
                token=token,
                max_tokens=max_tokens,
                fallback_max_tokens=fallback_max_tokens,
                temperature=temperature,
                timeout=timeout,
                retries=retries,
                docs_root=docs_root,
            ): index
            for index, row in ((index, row) for index, row in enumerate(rows) if index not in preserved)
        }
        with tqdm(total=len(rows_to_evaluate), desc="bfcl", unit="sample", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results_by_index[index] = result
                bar.update(1)

    missing = [index for index in range(len(rows)) if index not in results_by_index]
    if missing:
        raise RuntimeError(f"missing results after evaluation: {missing[:5]}")
    results = [results_by_index[index] for index in range(len(rows))]

    summary = summarize_rows(results)
    completed = sum(1 for row in results if row and row.get("completed"))
    incomplete = len(rows) - completed
    reward = summary["accuracy"] if incomplete == 0 else None
    meta_data = {
        "benchmark": BENCHMARK_ID,
        "benchmark_id": BENCHMARK_ID,
        "model": model,
        "metric": "accuracy",
        "split": split,
        "total": len(rows),
        "scoreable_total": summary["scoreable_total"],
        "scoreable_correct": summary["scoreable_correct"],
        "scoreable_completed": summary["scoreable_completed"],
        "unscored_total": summary["unscored_total"],
        "completed": completed,
        "incomplete": incomplete,
        "accuracy": summary["accuracy"],
        "truncated_total": summary["truncated_total"],
        "elapsed_s": round(time.time() - started, 3),
        "provider": "openai-compatible",
        "score_notes": "Reward excludes rest rows because the public BFCL release does not ship a released answer key for that subset. Token/context-truncated scoreable rows receive zero credit.",
        "category_stats": summary["category_stats"],
    }

    (output_dir / "samples.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, default=lambda value: "..." if value is Ellipsis else str(value)) for row in results) + "\n", encoding="utf-8")
    (output_dir / "reward.json").write_text(json.dumps({"reward": reward, "meta_data": meta_data}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(resolved_config(config, model, split, output_dir), allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({"benchmark": BENCHMARK_ID, "status": "ok" if incomplete == 0 else "incomplete", "reward": reward, "accuracy": summary["accuracy"], "completed": completed, "incomplete": incomplete, "truncated_total": summary["truncated_total"], "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    if incomplete:
        raise SystemExit(f"error: {incomplete} incomplete BFCL samples")


if __name__ == "__main__":
    main()

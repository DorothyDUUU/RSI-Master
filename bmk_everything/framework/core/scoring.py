from __future__ import annotations

import json
import math
import re
import unicodedata


FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:：]\s*(.+)", re.I)
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def normalize(text: object) -> str:
    # `str(text or "")` collapses falsy values (False, 0, "") to "", which made
    # boolean-False and numeric-zero targets unmatchable. Only None should map
    # to "".
    value = unicodedata.normalize("NFKC", "" if text is None else str(text)).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(".。")


def final_text(text: str) -> str:
    matches = FINAL_RE.findall(text or "")
    return matches[-1].strip() if matches else (text or "").strip()


def exact_match(prediction: str, targets: list[object]) -> bool:
    pred = normalize(final_text(prediction))
    return any(pred == normalize(target) for target in targets)


def boolean_match(prediction: str, targets: list[object]) -> bool:
    aliases = {
        "true": "yes", "yes": "yes", "1": "yes",
        "false": "no", "no": "no", "0": "no",
    }
    pred = normalize(final_text(prediction)).split(maxsplit=1)[0].rstrip(".,:;!?")
    expected = {aliases.get(normalize(target), normalize(target)) for target in targets}
    return aliases.get(pred, pred) in expected


def label_match(prediction: str, targets: list[object]) -> bool:
    pred = normalize(final_text(prediction))
    for target in targets:
        label = normalize(target)
        if pred == label or re.match(rf"^{re.escape(label)}(?:\W|$)", pred):
            return True
    return False


def squad_match(prediction: str, targets: list[object]) -> bool:
    def canonical(value: object) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    pred = canonical(final_text(prediction))
    expected = {canonical(target) for target in targets}
    no_answer_targets = {"unanswerable", "no answer"}
    no_answer_predictions = {
        "unanswerable", "no answer", "not answerable", "cannot be answered",
        "passage does not say", "not stated in passage", "not mentioned in passage",
        "information is not provided", "information not provided",
    }
    if expected & no_answer_targets and pred in no_answer_predictions:
        return True
    return pred in expected


def unscored(prediction: str, targets: list[object]) -> None:
    return None


def numeric_match(prediction: str, targets: list[object], tolerance: float = 1e-6) -> bool:
    values = NUMBER_RE.findall(final_text(prediction).replace(",", ""))
    if not values:
        return False
    try:
        pred = float(values[-1])
    except ValueError:
        return False
    for target in targets:
        match = NUMBER_RE.search(str(target).replace(",", ""))
        if match:
            expected = float(match.group())
            if math.isclose(pred, expected, rel_tol=tolerance, abs_tol=tolerance):
                return True
    return False


def json_match(prediction: str, targets: list[object]) -> bool:
    text = final_text(prediction)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        return False
    try:
        pred = json.loads(text[start:])
    except json.JSONDecodeError:
        return False
    for target in targets:
        try:
            expected = target if not isinstance(target, str) else json.loads(target)
        except json.JSONDecodeError:
            continue
        if pred == expected:
            return True
    return False


def multiple_choice_match(prediction: str, targets: list[object]) -> bool:
    text = final_text(prediction).strip()
    letter = re.search(r"(?:^|\b)([A-Z])(?:\b|$)", text.upper())
    candidates = [normalize(x) for x in targets]
    if letter and normalize(letter.group(1)) in candidates:
        return True
    return normalize(text) in candidates

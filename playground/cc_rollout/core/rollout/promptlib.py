"""Prompt template loader — every prompt lives in ``rollout/prompt/``.

移植自 ``playground/cc_branching/core/rollout/promptlib.py``, 唯一区别是
:func:`compose` 接受**协议文件名**而不是 branching/single 特判 —— cc_rollout 的
两种模式 (``protocol_naive.md`` / ``protocol_with_expos.md``) 由 seed 层选择。

Agents never get prompts assembled inline in code: the text is always a file
under ``rollout/prompt/`` (``protocol_naive.md``, ``protocol_with_expos.md``,
``shared_context.md``), and placeholders of the form ``{name}`` are filled by
:func:`fill`.

``fill`` substitutes in a **single pass** over a regex alternation of the given
placeholder names, so a value inserted early can never be re-interpreted later
(e.g. an agent's ``AGENT.md`` containing the literal text ``{n}`` stays intact).

:func:`compose` builds the workspace README (the agent's system prompt): a role
protocol file plus the shared context, with the ``{backend_descriptions}``
placeholder in the shared context filled from the seeded backend modules (see
:mod:`.backend_readmes`).
"""

from __future__ import annotations

import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompt"

#: Neutral replacement when no backend descriptions were injected — a literal
#: placeholder must never reach the agent.
_NO_BACKEND_DESCRIPTIONS = "(no backend modules were selected for this run.)"
#: Neutral replacement when no eval contract was injected (no bench_id in config).
_NO_EVAL_CONTRACT = "(no eval contract — bench_id is unset in this run's config.)"
#: Neutral replacement when the mode-specific submit driver wasn't injected.
_NO_SUBMIT_DRIVER = (
    "(submit driver not specified for this run — see your protocol for how to "
    "launch a training pipeline.)"
)
#: Neutral replacement when no run-window (wall-clock budget) was injected.
_NO_RUN_WINDOW = (
    "(no run window was injected — check the run config's "
    "`run.single_agent_running_time_seconds` and read the current time with "
    "`date -u` to pace yourself.)"
)


def load(name: str) -> str:
    """Read a prompt template file verbatim."""
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def fill(name: str, **placeholders: object) -> str:
    """Load *name* and replace ``{key}`` with each *placeholders* value.

    Unknown ``{...}`` sequences in the template are left untouched.
    """
    text = load(name)
    if not placeholders:
        return text
    alt = "|".join(re.escape("{" + k + "}") for k in placeholders)
    return re.sub(alt, lambda m: str(placeholders[m.group(0)[1:-1]]), text)


def compose(
    protocol: str,
    backend_descriptions: str = "",
    eval_contract: str = "",
    submit_driver: str = "",
    run_window: str = "",
) -> str:
    """Compose a workspace README: role protocol + shared context.

    Args:
        protocol: the protocol template filename under ``prompt/`` (e.g.
            ``"protocol_naive.md"`` or ``"protocol_with_expos.md"``).
        backend_descriptions: the rendered ``BACKEND_DESCRIPTIONS`` block from
            :func:`.backend_readmes.build_backend_descriptions`; replaces the
            ``{backend_descriptions}`` placeholder in the shared context.
            When empty, a neutral default is substituted so no literal
            ``{...}`` ever reaches the agent.
        eval_contract: the run's eval contract (bench_id / limit / serve args)
            injected under the Run Contract section; empty → a neutral default
            so no literal ``{...}`` reaches the agent.
        submit_driver: the mode-specific "how to drive the submit-train
            pipeline" block (cc_naive → ``public/pipeline-submit``; cc_with_expos
            → ``./expos worker submit`` + SUBMIT_GUIDE.md), replacing the
            ``{submit_driver}`` placeholder in the shared context.
        run_window: the wall-clock run window (start / end / budget) rendered by
            the seed layer, replacing the ``{run_window}`` placeholder in the
            shared context.  Empty → a neutral default so no literal ``{...}``
            ever reaches the agent.

    The protocol file carries the research methodology (incl. the mode-specific
    workspace layout); the shared context (backend modules, launching-training
    advice, the submit driver, leakage, time discipline, eval contract, process
    discipline) is appended after a ``---`` separator.  Placeholders are filled
    in a single pass — any ``{...}`` sequences inside the injected text are
    never re-parsed.
    """
    proto = load(protocol)
    shared = fill("shared_context.md",
                  backend_descriptions=backend_descriptions or _NO_BACKEND_DESCRIPTIONS,
                  eval_contract=eval_contract or _NO_EVAL_CONTRACT,
                  submit_driver=submit_driver or _NO_SUBMIT_DRIVER,
                  run_window=run_window or _NO_RUN_WINDOW)
    return proto.rstrip() + "\n\n---\n\n" + shared.rstrip() + "\n"

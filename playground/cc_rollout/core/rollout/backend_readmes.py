"""Backend README summaries injected into the composed agent system prompt.

逐字移植自 ``playground/cc_branching/core/rollout/backend_readmes.py``。种子层把
选中的 backend 模块拷贝进 ``workspace/public/`` (见 ``seed._seed_public_modules``),
这里读取**实际被 seed** 的每个模块的 workspace 副本里的 ``BACKEND_DESCRIPTIONS``
摘要块, 渲染一段紧凑的说明 (摘要 + workspace 相对路径), 交给 ``promptlib.compose``
注入 workspace README (agent 的 system prompt)。

读 workspace 副本而不是源码树, 保证路径就是 agent 要 ``cat`` 的路径, 也天然跟随
eval 漏斗: 被 ``eval.backend`` 过滤掉的模块不在副本里, 也就不会被描述。

Pure functions, no orchestration imports.  Best-effort: a missing README or
missing block degrades to a path-only entry, never a crash.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BACKEND_DESCRIPTIONS_MARKER = "BACKEND_DESCRIPTIONS"

#: Module name → (workspace-relative directory, workspace-relative README path).
#: Paths match ``_seed_public_modules``'s layout: gpu_backend lands in
#: ``public/gpu_backend/``, backend/backend/* modules in ``public/backend/<name>/``.
MODULE_README_RELPATH: dict[str, tuple[str, str]] = {
    "gpu_backend": ("public/gpu_backend/", "public/gpu_backend/README.md"),
    "eval_all": ("public/backend/eval_all/", "public/backend/eval_all/README.md"),
    "slime": ("public/backend/slime/", "public/backend/slime/README.md"),
    "search_cli": ("public/backend/search_cli/", "public/backend/search_cli/README.md"),
    "submit_pipeline": ("public/backend/submit_pipeline/", "public/backend/submit_pipeline/README.md"),
    "vllm": ("public/backend/vllm/", "public/backend/vllm/README.md"),
}

#: Fallback line when a README has no BACKEND_DESCRIPTIONS block.
_NO_BLOCK_FALLBACK = "(no BACKEND_DESCRIPTIONS block in README; see the file for details)"
#: Placeholder when no modules were seeded this run.
_NONE_SELECTED = "(none selected this run)"


def extract_backend_description(text: str) -> str | None:
    """Extract the ``BACKEND_DESCRIPTIONS`` block content from a README.

    Expected layout (block near the top of the file)::

        ---
        BACKEND_DESCRIPTIONS
        <summary>
        ---

    Returns the summary (whitespace-stripped), or ``None`` when there is no
    marker, no closing fence, or the block body is empty.
    """
    if text is None:
        return None
    lines = text.splitlines()
    marker_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip() == BACKEND_DESCRIPTIONS_MARKER),
        None,
    )
    if marker_idx is None:
        return None
    # Block is fenced with "---" above and below the marker.  If the marker is
    # the very first line there is no opening fence; body then starts after it.
    open_idx = marker_idx
    if marker_idx > 0 and lines[marker_idx - 1].strip() == "---":
        open_idx = marker_idx - 1
    close_idx = next(
        (i for i in range(marker_idx + 1, len(lines)) if lines[i].strip() == "---"),
        None,
    )
    if close_idx is None:
        return None
    body = [
        ln
        for ln in lines[open_idx + 1 : close_idx]
        if ln.strip() != BACKEND_DESCRIPTIONS_MARKER
    ]
    body_text = "\n".join(body).strip()
    return body_text or None


def _read_readme(workspace_root: Path, rel_readme: str) -> str | None:
    """Read ``workspace_root / rel_readme``; missing → ``None`` (never raises)."""
    try:
        path = workspace_root / rel_readme
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # defensive
        logger.warning("backend README read failed %s: %s", rel_readme, exc)
        return None


def build_backend_descriptions(seeded_modules: list[str], workspace_root: Path) -> str:
    """Render the ``BACKEND_DESCRIPTIONS`` prompt section for seeded modules.

    Args:
        seeded_modules: bare module names actually copied by
            ``_seed_public_modules`` (e.g. ``["gpu_backend", "eval_all", ...]``),
            in config order.
        workspace_root: the workspace dir (parent of ``public/``).

    Returns a fenced block like::

        ---
        BACKEND_DESCRIPTIONS
        ### eval_all
        - workspace: `public/backend/eval_all/` (README: `public/backend/eval_all/README.md`)
        <summary>

        ### gpu_backend
        ...
        ---

    Unknown module names are skipped with a warning; missing READMEs emit a
    path-only entry.  Empty ``seeded_modules`` returns ``(none selected this run)``.
    """
    entries: list[str] = []
    for module in seeded_modules:
        rel = MODULE_README_RELPATH.get(str(module))
        if rel is None:
            logger.warning(
                "backend_readmes: no README mapping for seeded module %r; skipping.",
                module,
            )
            continue
        rel_dir, rel_readme = rel
        header = (
            f"### {module}\n"
            f"- workspace: `{rel_dir}` (README: `{rel_readme}`)"
        )
        text = _read_readme(workspace_root, rel_readme)
        summary = extract_backend_description(text) if text is not None else None
        entries.append(f"{header}\n{summary if summary else _NO_BLOCK_FALLBACK}")

    if not entries:
        body = _NONE_SELECTED
    else:
        body = "\n\n".join(entries)
    return f"---\n{BACKEND_DESCRIPTIONS_MARKER}\n{body}\n---"


__all__ = [
    "BACKEND_DESCRIPTIONS_MARKER",
    "MODULE_README_RELPATH",
    "build_backend_descriptions",
    "extract_backend_description",
]

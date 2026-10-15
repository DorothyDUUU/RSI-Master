"""Render the ExperimentOS DAG (workers / experiments / reviewers) as a static HTML report.

Scans a run dir's ``experiment_os/state/state.json`` plus on-disk artifacts
(``experiments/*/result.json``, worker train yamls, ``reviews/*/reports``) and
emits a **self-contained static HTML** file containing:

- an inline-SVG layered DAG — worker tree, experiment nodes, reviewer nodes and
  their edges (worker→experiment, worker→reviewer, reviewer→experiment, parent
  worker→child worker)
- status / summary cards (loop, counters, best)
- detail tables: experiments (state score **vs** actual ``result.json`` score +
  orphan flag + training config), workers (objective), reviewers (verdict),
  data pool

Design notes:

- The score written into ``state.json`` is best-effort (see the known v2 gap
  where finished ``result.json`` files stay ``running/None``). This report
  therefore always cross-checks against ``experiments/{id}/result.json`` and
  flags orphans explicitly.
- No JS / no CDN: positions are computed here in Python and the HTML is a
  single static file (SVG + CSS). Tooltips use the native ``<title>`` element.

Usage:

    PYTHONPATH=tools/ExpOS python -m expos.dag_report \\
        --root runs/rsi_master_gsm8k_live \\
        --output dag_report.html

``--root`` may point at the run dir (which contains ``workspace/experiment_os``)
or directly at an ``experiment_os`` dir.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml optional
    yaml = None

# ── Layout constants (px) ────────────────────────────────────────────────────
Y0 = 40            # top of the first worker row
ROW_H = 240        # vertical spacing between worker tree levels
COL_GAP = 28       # horizontal gap between sibling worker subtrees
WORKER_W, WORKER_H = 150, 52
EX_W, EX_H = 112, 58
EX_GAP = 14
RV_W, RV_H = 132, 50
RV_GAP = 14
MARGIN = 40
MIN_BLOCK = WORKER_W + 40

# ── Status → style ───────────────────────────────────────────────────────────
_STYLE = {
    # status group: (fill, stroke)
    "ok":    ("#e8f5e9", "#2e7d32"),
    "warn":  ("#fff8e1", "#f9a825"),
    "bad":   ("#fdecea", "#c62828"),
    "gray":  ("#f5f5f5", "#9e9e9e"),
}
_BEST_STROKE = "#f9a825"


def _status_group(status: str | None) -> str:
    s = (status or "").lower()
    if s in {"finished", "completed"}:
        return "ok"
    if s in {"running", "pending", "queued", "in_progress", "active"}:
        return "warn"
    if s in {"failed", "aborted", "error"}:
        return "bad"
    return "gray"


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _find_state_root(root: Path) -> Path | None:
    """Locate ``state/state.json`` starting from a run dir or experiment_os dir."""
    for candidate in (
        root / "state" / "state.json",
        root / "workspace" / "experiment_os" / "state" / "state.json",
        root / "experiment_os" / "state" / "state.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _read_json(path: Path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _read_yaml(path: Path):
    if yaml is None:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _format_score(score: Any) -> str:
    if score is None:
        return "—"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def _short_objective(obj: str | None, limit: int = 220) -> str:
    if not obj:
        return ""
    obj = " ".join(obj.split())
    return obj if len(obj) <= limit else obj[: limit - 1] + "…"


def load_state(state_path: Path) -> dict[str, Any]:
    state = _read_json(state_path, {})
    if not state:
        raise SystemExit(f"could not read {state_path}")
    return state


# ── Data assembly ────────────────────────────────────────────────────────────


class WorkerBlock:
    def __init__(self, wid: str, node: dict[str, Any]):
        self.wid = wid
        self.node = node
        self.parent = node.get("parent_worker_id")
        self.children = list(node.get("child_worker_ids") or [])
        self.exp_ids = list(node.get("experiment_ids") or [])
        self.rev_ids = list(node.get("reviewer_ids") or [])
        self.depth = 0
        self.x = 0.0
        self.y = 0.0
        self.width = MIN_BLOCK
        self.objective = node.get("objective", "")


class DagReport:
    def __init__(self, root: Path):
        self.state_path = _find_state_root(root)
        if self.state_path is None:
            raise SystemExit(
                f"no state/state.json found under {root} (expected a run dir or "
                "an experiment_os dir)"
            )
        self.os_root = self.state_path.parent.parent  # …/experiment_os
        self.run_root = root
        self.state = load_state(self.state_path)

        g = self.state.get("graph", {})
        self.workers: dict[str, WorkerBlock] = {
            wid: WorkerBlock(wid, node) for wid, node in (g.get("workers") or {}).items()
        }
        self.experiments = g.get("experiments") or {}
        self.reviewers = g.get("reviewers") or {}
        self.root_ids = list(g.get("root_worker_ids") or [])
        self.loop = self.state.get("loop") or {}
        self.counters = self.state.get("counters") or {}
        self.best = self.state.get("best") or {}
        self.data_pool = self.state.get("data_pool_index") or {}

        self._assign_depths()
        self._load_experiment_details()
        self._load_reviewer_details()

    # ── tree geometry ────────────────────────────────────────────────────────

    def _roots(self) -> list[WorkerBlock]:
        if self.root_ids:
            return [self.workers[w] for w in self.root_ids if w in self.workers]
        return [w for w in self.workers.values() if not w.parent]

    def _assign_depths(self) -> None:
        for root in self._roots():
            stack = [(root, 0)]
            while stack:
                node, depth = stack.pop()
                if depth > node.depth:
                    node.depth = depth
                for cid in node.children:
                    child = self.workers.get(cid)
                    if child is not None:
                        stack.append((child, depth + 1))
        # workers not reachable from any root (defensive)
        for w in self.workers.values():
            if not self._roots() and w.depth == 0:
                pass

    def _subtree_width(self, w: WorkerBlock) -> float:
        exp_row = max(0, len(w.exp_ids) * (EX_W + EX_GAP) - EX_GAP)
        rev_row = max(0, len(w.rev_ids) * (RV_W + RV_GAP) - RV_GAP)
        child_sum = sum(self._subtree_width(self.workers[c]) for c in w.children if c in self.workers)
        if w.children:
            child_sum += (len(w.children) - 1) * COL_GAP
        return max(exp_row, rev_row, child_sum, MIN_BLOCK)

    def _place(self, w: WorkerBlock, x0: float) -> float:
        """Place worker w and its subtree; return the x just past the subtree."""
        w.width = self._subtree_width(w)
        w.x = x0 + w.width / 2 - WORKER_W / 2
        w.y = Y0 + w.depth * ROW_H
        cx = x0
        for cid in w.children:
            child = self.workers.get(cid)
            if child is None:
                continue
            cw = self._subtree_width(child)
            self._place(child, cx)
            cx += cw + COL_GAP
        return x0 + w.width

    def _layout(self) -> None:
        cx = MARGIN
        for root in self._roots():
            cx = self._place(root, cx) + COL_GAP
        self.width = max(MARGIN + WORKER_W + MARGIN, cx + MARGIN)

        # experiment / reviewer node positions (below their worker)
        self.exp_pos: dict[str, tuple[float, float]] = {}
        self.rev_pos: dict[str, tuple[float, float]] = {}
        for w in self.workers.values():
            if w.exp_ids:
                row_w = len(w.exp_ids) * (EX_W + EX_GAP) - EX_GAP
                start = w.x + WORKER_W / 2 - row_w / 2
                for i, eid in enumerate(w.exp_ids):
                    self.exp_pos[eid] = (start + i * (EX_W + EX_GAP), w.y + WORKER_H + 18)
            if w.rev_ids:
                row_w = len(w.rev_ids) * (RV_W + RV_GAP) - RV_GAP
                start = w.x + WORKER_W / 2 - row_w / 2
                for i, rid in enumerate(w.rev_ids):
                    self.rev_pos[rid] = (start + i * (RV_W + RV_GAP), w.y + WORKER_H + 18 + EX_H + 26)

        bottom = max((w.y for w in self.workers.values()), default=Y0)
        self.height = bottom + WORKER_H + 18 + EX_H + 26 + RV_H + MARGIN

    # ── detail enrichment ────────────────────────────────────────────────────

    def _load_experiment_details(self) -> None:
        for eid, node in self.experiments.items():
            edir = self.os_root / "experiments" / eid
            result = _read_json(edir / "result.json", {}) or {}
            node["_result"] = result
            node["_result_score"] = result.get("score")
            node["_result_status"] = result.get("status")
            node["_train_cfg"] = {}
            tc_path = (result.get("effective_config") or {}).get("training_config_path")
            if tc_path:
                tc = Path(tc_path)
                if not tc.is_absolute():
                    tc = self.os_root / tc
                node["_train_cfg"] = _read_yaml(tc)
            # orphan: finished/failed on disk but state still running/pending/None
            rs = (result.get("status") or "").lower()
            ss = (node.get("status") or "").lower()
            node["_orphan"] = rs in {"completed", "failed"} and ss in {"running", "pending", ""}

    def _load_reviewer_details(self) -> None:
        for rid, node in self.reviewers.items():
            rdir = self.os_root / "reviews" / rid
            reports: list[dict] = []
            for exp_id in node.get("reviewed_experiment_ids") or []:
                rep = _read_json(rdir / "reports" / f"{exp_id}.json", None)
                if rep is not None:
                    reports.append(rep)
            node["_reports"] = reports

    # ── SVG rendering ────────────────────────────────────────────────────────

    def _node_rect(self, x: float, y: float, w: float, h: float, status: str | None,
                   label: str, sub: str, tooltip: str, *, best: bool = False) -> list[str]:
        fill, stroke = _STYLE[_status_group(status)]
        stroke_w = 3 if best else 1.5
        stroke = _BEST_STROKE if best else stroke
        parts = [
            f'  <g>',
            f'    <title>{_esc(tooltip)}</title>',
            f'    <rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="6" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>',
            f'    <text x="{x + w / 2:.1f}" y="{y + h / 2 - 1:.1f}" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#1a1a1a">{_esc(label)}</text>',
        ]
        if sub:
            parts.append(
                f'    <text x="{x + w / 2:.1f}" y="{y + h / 2 + 13:.1f}" text-anchor="middle" '
                f'font-size="10" fill="#555">{_esc(sub)}</text>'
            )
        parts.append("  </g>")
        return parts

    def _edge(self, x1: float, y1: float, x2: float, y2: float, *, curve: bool = False,
              color: str = "#9aa0a6", dash: bool = False) -> str:
        dash_attr = ' stroke-dasharray="5,4"' if dash else ""
        if curve:
            mid = (y1 + y2) / 2
            d = (f"M {x1:.1f} {y1:.1f} C {x1:.1f} {mid:.1f}, "
                 f"{x2:.1f} {mid:.1f}, {x2:.1f} {y2:.1f}")
        else:
            d = f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}"
        return (f'  <path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{dash_attr} '
                f'marker-end="url(#arrow)"/>')

    def render_svg(self) -> str:
        self._layout()
        parts: list[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width:.0f} {self.height:.0f}" '
            f'width="100%" style="max-width:{self.width}px;height:auto;background:#fff;'
            f'border:1px solid #e0e0e0;border-radius:8px">'
        )
        parts.append(
            '  <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
            'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#9aa0a6"/></marker></defs>'
        )

        # edges: parent→child worker
        for w in self.workers.values():
            for cid in w.children:
                c = self.workers.get(cid)
                if c is None:
                    continue
                parts.append(self._edge(w.x + WORKER_W / 2, w.y + WORKER_H,
                                        c.x + WORKER_W / 2, c.y))
        # edges: worker→experiment, worker→reviewer
        for w in self.workers.values():
            cx = w.x + WORKER_W / 2
            for eid in w.exp_ids:
                ex, ey = self.exp_pos[eid]
                parts.append(self._edge(cx, w.y + WORKER_H, ex + EX_W / 2, ey))
            for rid in w.rev_ids:
                rx, ry = self.rev_pos[rid]
                parts.append(self._edge(cx, w.y + WORKER_H, rx + RV_W / 2, ry))
        # edges: reviewer→reviewed experiment (may span columns)
        for rid, node in self.reviewers.items():
            if rid not in self.rev_pos:
                continue
            rx, ry = self.rev_pos[rid]
            for eid in node.get("reviewed_experiment_ids") or []:
                if eid in self.exp_pos:
                    ex, ey = self.exp_pos[eid]
                    parts.append(self._edge(rx + RV_W / 2, ry + RV_H, ex + EX_W / 2, ey + EX_H / 2,
                                            curve=True, color="#b39ddb", dash=True))

        # nodes
        for w in sorted(self.workers.values(), key=lambda x: x.wid):
            best_w = self.best.get("worker_id") == w.wid
            tooltip = (
                f"{w.wid} · status={w.node.get('status')} · best_score="
                f"{_format_score(w.node.get('best_score'))}\n\n"
                f"objective: {_short_objective(w.objective, 800)}"
            )
            parts += self._node_rect(
                w.x, w.y, WORKER_W, WORKER_H, w.node.get("status"),
                w.wid,
                f"best={_format_score(w.node.get('best_score'))}" if w.node.get("best_score") is not None else "",
                tooltip,
                best=best_w,
            )
        for eid, node in sorted(self.experiments.items(), key=lambda x: x[0]):
            ex, ey = self.exp_pos.get(eid, (-9999, -9999))
            if ex < 0:
                continue
            best_e = self.best.get("experiment_id") == eid
            cfg = node.get("_train_cfg", {})
            cfg_line = ", ".join(
                f"{k}={v}" for k, v in (
                    ("epochs", cfg.get("num_train_epochs")),
                    ("samples", cfg.get("max_samples")),
                    ("steps", cfg.get("max_steps")),
                    ("lr", cfg.get("learning_rate")),
                ) if v is not None
            )
            tooltip = (
                f"{eid} · worker={node.get('worker_id')} · status={node.get('status')} · "
                f"state_score={_format_score(node.get('score'))}\n"
                f"result.json: status={node.get('_result_status')} "
                f"score={_format_score(node.get('_result_score'))}"
                f"{'  ⚠ ORPHAN (result finished but state not synced)' if node.get('_orphan') else ''}\n"
                f"mode={node.get('training_mode')} · benchmark={node.get('benchmark')}\n"
                f"train_cfg: {cfg_line or '(n/a)'}\n"
                f"dir: {node.get('workspace')}"
            )
            parts += self._node_rect(
                ex, ey, EX_W, EX_H, node.get("status"),
                eid,
                _format_score(node.get("score")),
                tooltip,
                best=best_e,
            )
        for rid, node in sorted(self.reviewers.items(), key=lambda x: x[0]):
            rx, ry = self.rev_pos.get(rid, (-9999, -9999))
            if rx < 0:
                continue
            verdict = node.get("verdict_summary", {}) or {}
            v = verdict.get("verdict") or "—"
            tooltip = (
                f"{rid} · worker={node.get('worker_id')} · status={node.get('status')}\n"
                f"reviews: {', '.join(node.get('reviewed_experiment_ids') or [])}\n"
                f"verdict: {_esc(v)}\n"
                f"reports: {len(node.get('_reports', []))} file(s)"
            )
            parts += self._node_rect(
                rx, ry, RV_W, RV_H, node.get("status"),
                rid,
                f"verdict={v}",
                tooltip,
            )
        parts.append("</svg>")
        return "\n".join(parts)

    # ── tables ───────────────────────────────────────────────────────────────

    def experiments_table(self) -> str:
        rows = []
        for eid, node in sorted(self.experiments.items(), key=lambda x: x[0]):
            cfg = node.get("_train_cfg", {})
            cfg_line = ", ".join(
                f"{k}={v}" for k, v in (
                    ("epochs", cfg.get("num_train_epochs")),
                    ("samples", cfg.get("max_samples")),
                    ("steps", cfg.get("max_steps")),
                    ("lr", cfg.get("learning_rate")),
                ) if v is not None
            ) or "—"
            orphan = node.get("_orphan")
            rows.append(
                "<tr>"
                f"<td>{_esc(eid)}</td>"
                f"<td>{_esc(node.get('worker_id'))}</td>"
                f"<td>{_esc(node.get('training_mode'))}</td>"
                f"<td>{_esc(node.get('status'))}</td>"
                f"<td>{_format_score(node.get('score'))}</td>"
                f"<td>{_esc(node.get('_result_status'))}</td>"
                f"<td>{_format_score(node.get('_result_score'))}</td>"
                f"<td>{'<b style=color:#c62828>⚠ orphan</b>' if orphan else ''}</td>"
                f"<td>{_esc(cfg_line)}</td>"
                "</tr>"
            )
        return self._table(
            ["experiment", "worker", "mode", "state status", "state score",
             "result status", "result score", "orphan", "train config"],
            rows,
        )

    def workers_table(self) -> str:
        rows = []
        for w in sorted(self.workers.values(), key=lambda x: x.wid):
            n = w.node
            rows.append(
                "<tr>"
                f"<td>{_esc(w.wid)}</td>"
                f"<td>{_esc(n.get('parent_worker_id') or 'root')}</td>"
                f"<td>{_esc(n.get('status'))}</td>"
                f"<td>{_format_score(n.get('best_score'))}</td>"
                f"<td>{_esc(n.get('best_experiment_id') or '—')}</td>"
                f"<td>{_esc(', '.join(w.exp_ids) or '—')}</td>"
                f"<td>{_esc(', '.join(w.rev_ids) or '—')}</td>"
                f"<td style='max-width:520px'>{_esc(_short_objective(w.objective, 300))}</td>"
                "</tr>"
            )
        return self._table(
            ["worker", "parent", "status", "best score", "best exp",
             "experiments", "reviewers", "objective"],
            rows,
        )

    def reviewers_table(self) -> str:
        rows = []
        for rid, node in sorted(self.reviewers.items(), key=lambda x: x[0]):
            verdict = node.get("verdict_summary", {}) or {}
            v = verdict.get("verdict") or "—"
            bottleneck = verdict.get("primary_bottleneck") or "—"
            rows.append(
                "<tr>"
                f"<td>{_esc(rid)}</td>"
                f"<td>{_esc(node.get('worker_id'))}</td>"
                f"<td>{_esc(', '.join(node.get('reviewed_experiment_ids') or []))}</td>"
                f"<td>{_esc(node.get('status'))}</td>"
                f"<td>{_esc(str(v))}</td>"
                f"<td>{_esc(_short_objective(str(bottleneck), 160))}</td>"
                f"<td>{len(node.get('_reports', []))}</td>"
                "</tr>"
            )
        return self._table(
            ["reviewer", "worker", "audits", "status", "verdict", "primary bottleneck", "reports"],
            rows,
        )

    def data_pool_table(self) -> str:
        pool = self.data_pool
        if not pool:
            return "<p class='muted'>no data pool index in state.json</p>"
        rows = []
        if isinstance(pool, dict):
            items = pool.get("entries") if isinstance(pool, dict) and "entries" in pool else pool
            items = items.items() if isinstance(items, dict) else items
            for data_id, rec in items:
                if isinstance(rec, str):
                    rows.append(
                        "<tr><td>%s</td><td colspan=3 class=muted>%s</td></tr>"
                        % (_esc(data_id), _esc(rec))
                    )
                    continue
                rows.append(
                    "<tr>"
                    f"<td>{_esc(data_id)}</td>"
                    f"<td>{_esc(rec.get('source_worker_id') or '—')}</td>"
                    f"<td>{_esc(rec.get('artifact_type') or '—')}</td>"
                    f"<td>{_esc(rec.get('num_rows') or '—')}</td>"
                    f"<td>{_esc(rec.get('promotion_status') or '—')}</td>"
                    "</tr>"
                )
        return self._table(["data_id", "source worker", "type", "rows", "promotion"], rows)

    @staticmethod
    def _table(headers: list[str], rows: list[str]) -> str:
        thead = "".join(f"<th>{_esc(h)}</th>" for h in headers)
        body = "\n".join(rows) if rows else "<tr><td colspan=99 class=muted>—</td></tr>"
        return (
            "<table><thead><tr>"
            + thead
            + "</tr></thead><tbody>"
            + body
            + "</tbody></table>"
        )

    # ── HTML shell ───────────────────────────────────────────────────────────

    def render_html(self) -> str:
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        best = self.best or {}
        loop_status = self.loop.get("status", "?")
        stop_reason = self.loop.get("stop_reason")
        counters = self.counters
        cards = [
            ("loop", f"{loop_status}" + (f"<br><span class=muted>{_esc(stop_reason)}</span>" if stop_reason else "")),
            ("workers", str(counters.get("worker", "?"))),
            ("experiments", str(counters.get("experiment", "?"))),
            ("reviewers", str(counters.get("reviewer", "?"))),
            ("data", str(counters.get("data", "?"))),
            ("best", f"{_esc(best.get('experiment_id'))} · {_format_score(best.get('score'))}"),
        ]
        cards_html = "".join(
            f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{v}</div></div>'
            for k, v in cards
        )
        task = self.state.get("task_description", "")

        body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ExperimentOS DAG report — {_esc(self.run_root.name)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
         margin: 24px; color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  h2 {{ font-size: 15px; margin: 28px 0 8px; border-bottom: 2px solid #eee; padding-bottom: 4px; }}
  .muted {{ color: #888; font-weight: normal; }}
  .sub {{ color: #666; font-size: 12px; margin-bottom: 16px; }}
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
  .card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px 16px; min-width: 96px; }}
  .card .k {{ font-size: 11px; text-transform: uppercase; color: #888; }}
  .card .v {{ font-size: 18px; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 12px; }}
  th, td {{ border: 1px solid #e5e5e5; padding: 5px 8px; text-align: left; }}
  th {{ background: #f3f4f6; position: sticky; top: 0; }}
  tr:nth-child(even) {{ background: #fafbfc; }}
  .wrap {{ overflow-x: auto; margin-top: 16px; }}
  .task {{ background: #fff; border-left: 3px solid #1976d2; padding: 10px 14px; font-size: 13px; }}
</style>
</head>
<body>
<h1>ExperimentOS DAG — {_esc(self.run_root.name)}</h1>
<div class="sub">
  state: <code>{_esc(str(self.state_path))}</code> · generated {_esc(now)}
</div>
<div class="task">{_esc(task) or '<span class=muted>(no task_description)</span>'}</div>
<div class="cards">{cards_html}</div>
<h2>DAG (workers → experiments → reviewers)</h2>
<div class="wrap">{self.render_svg()}</div>
<h2>Experiments</h2>
{self.experiments_table()}
<h2>Workers</h2>
{self.workers_table()}
<h2>Reviewers</h2>
{self.reviewers_table()}
<h2>Data pool</h2>
{self.data_pool_table()}
</body>
</html>
"""
        return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the ExperimentOS DAG as a static HTML report")
    parser.add_argument("--root", required=True, help="run dir or experiment_os dir to scan")
    parser.add_argument("--output", default=None, help="output HTML path (default: <root>_dag_report.html)")
    parser.add_argument("--open", action="store_true", help="try to open the report after writing")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    report = DagReport(root)
    output = Path(args.output) if args.output else Path(f"{root}_dag_report.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.render_html(), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    print(f"  workers={len(report.workers)} experiments={len(report.experiments)} "
          f"reviewers={len(report.reviewers)}")
    if args.open:
        import webbrowser
        webbrowser.open(output.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build a single self-contained HTML report of all 200 benchmarks.

For every benchmark it shows: id/name, domain, scorer, license, sample count,
score + completion (from the batch analysis JSON), and the full BENCHMARK.md
content (rendered to HTML) inside an expandable card.

Usage::

    framework/.venv/bin/python framework/automation/build_report_html.py \
      --analysis data/output/qwen36-batch/qwen36-20260808/analysis-final.json \
      --format-dir format --out data/output/qwen36-batch/qwen36-20260808/benchmarks_report.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- markdown → html

_INLINE_RE = [
    (re.compile(r"`([^`]+)`"), lambda m: f"<code>{html.escape(m.group(1))}</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), lambda m: f"<strong>{m.group(1)}</strong>"),
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), lambda m: f"<em>{m.group(1)}</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>'),
]


def _inline(text: str) -> str:
    for pat, repl in _INLINE_RE:
        text = pat.sub(repl, text)
    return text


def _table(lines: list[str]) -> str:
    rows = []
    for ln in lines:
        if re.match(r"^\s*\|?[\s:|-]+\|?\s*$", ln) and "-" in ln:
            continue  # separator row
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    out = ["<table><thead><tr>"]
    out += "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
    out += "</tr></thead><tbody>"
    for row in rows[1:]:
        out += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
    out += "</tbody></table>"
    return "".join(out)


def markdown_to_html(md: str) -> str:
    """Minimal markdown renderer covering the constructs used in BENCHMARK.md."""
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    para: list[str] = []
    list_items: list[str] = []
    list_type = None
    in_code = False
    code: list[str] = []

    def flush_para():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    def flush_list():
        nonlocal list_type
        if list_items:
            tag = "ol" if list_type == "ol" else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in list_items) + f"</{tag}>")
            list_items.clear()
            list_type = None

    while i < n:
        ln = lines[i]
        stripped = ln.strip()

        if stripped.startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                code.clear()
                in_code = False
            else:
                flush_para(); flush_list()
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(ln)
            i += 1
            continue

        if not stripped:
            flush_para(); flush_list()
            i += 1
            continue

        if stripped.startswith("#"):
            flush_para(); flush_list()
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("# ").strip()
            out.append(f"<h{level}>{_inline(text)}</h{level}>")
        elif stripped.startswith("|"):
            flush_para(); flush_list()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i]); i += 1
            out.append(_table(tbl))
            continue
        elif re.match(r"^\s*(?:[-*]|\d+[.)])\s+", ln):
            flush_para()
            m = re.match(r"^\s*(?:([-*])|\d+[.)])\s+(.*)$", ln)
            if m and m.group(1):
                if list_type != "ul":
                    flush_list(); list_type = "ul"
            else:
                if list_type != "ol":
                    flush_list(); list_type = "ol"
            list_items.append(m.group(2) if m else ln.strip())
        elif stripped.startswith(">"):
            flush_para(); flush_list()
            quote = stripped.lstrip(">").strip()
            out.append(f"<blockquote>{_inline(quote)}</blockquote>")
        elif stripped in ("---", "***", "___"):
            flush_para(); flush_list()
            out.append("<hr>")
        else:
            flush_list()
            para.append(ln)
        i += 1

    flush_para(); flush_list()
    if in_code:
        out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
    return "".join(out)


# ---------------------------------------------------------------- report build

def load_meta(format_dir: Path) -> dict:
    meta = {}
    for spec_path in sorted(format_dir.glob("*/spec.json")):
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        bid = spec.get("benchmark-id")
        if not bid:
            continue
        hf = spec.get("dataset", {}).get("hf") or {}
        meta[bid] = {
            "name": spec.get("benchmark-name") or bid,
            "domain": spec.get("domain") or "-",
            "scorer": spec.get("scorer") or "-",
            "license": (spec.get("license") or {}).get("spdx") or "-",
            "local_rows": (spec.get("dataset") or {}).get("local_rows"),
            "hf_path": hf.get("path") or "-",
            "source": spec.get("source") or "-",
            "adapter": spec.get("adapter") or "-",
        }
    for bid, dom, scorer in (("gsm8k", "math_reasoning", "exact_match"),
                             ("gpqa", "domain_knowledge", "multiple_choice"),
                             ("aime2025", "math_reasoning", "numeric")):
        meta.setdefault(bid, {"name": bid, "domain": dom, "scorer": scorer,
                              "license": "-", "local_rows": None,
                              "hf_path": "-", "source": "-", "adapter": "legacy_runner"})
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True,
                        help="summarize_batch.py JSON output")
    parser.add_argument("--format-dir", type=Path, default=REPO / "format")
    parser.add_argument("--out", type=Path,
                        default=REPO / "data/output/qwen36-batch/qwen36-20260808/benchmarks_report.html")
    parser.add_argument("--ids", type=str, default=None,
                        help="filter to these benchmark ids: comma-separated list, or a file "
                             "with one id per line (default: all)")
    parser.add_argument("--title", type=str, default=None,
                        help="override the report <h1> title (default derived from count)")
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    results = analysis["results"]
    if args.ids:
        ids_path = Path(args.ids)
        if ids_path.exists():
            id_set = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
        else:
            id_set = {item.strip() for item in args.ids.split(",") if item.strip()}
        filtered = [r for r in results if r["benchmark"] in id_set]
        print(f"[report] filter {len(results)} -> {len(filtered)} benchmarks "
              f"(missing from analysis: {sorted(id_set - {r['benchmark'] for r in results})})")
        results = filtered
    meta = load_meta(args.format_dir)

    cards = []
    for r in sorted(results, key=lambda x: x["benchmark"]):
        bid = r["benchmark"]
        m = meta.get(bid, {})
        score = r["score"]
        score_txt = f"{score:.4f}" if score is not None else "—"
        score_cls = ("s-high" if (score or 0) >= 0.9
                     else "s-mid" if (score or 0) >= 0.5
                     else "s-low" if score is not None else "s-none")
        complete = r["completed"] if r["completed"] is not None else 0
        attempted = r["attempted"] if r["attempted"] is not None else 0
        pct = f"{100 * complete / max(attempted, 1):.1f}%" if attempted else "—"
        md_path = args.format_dir / bid / "BENCHMARK.md"
        md_html = markdown_to_html(md_path.read_text()) if md_path.exists() else \
            "<p class='warn'>（无 BENCHMARK.md）</p>"
        cards.append({
            "id": bid, "name": m.get("name", bid), "domain": m.get("domain", "-"),
            "scorer": m.get("scorer", "-"), "license": m.get("license", "-"),
            "local_rows": m.get("local_rows"), "hf_path": m.get("hf_path", "-"),
            "source": m.get("source", "-"), "adapter": m.get("adapter", "-"),
            "score": score, "score_txt": score_txt, "score_cls": score_cls,
            "complete": complete, "attempted": attempted, "pct": pct,
            "failures": r.get("failures") or 0,
            "md_html": md_html,
        })

    dom_vals = sorted({c["domain"] for c in cards})
    scorer_vals = sorted({c["scorer"] for c in cards})

    total_a = sum(c["attempted"] for c in cards)
    total_c = sum(c["complete"] for c in cards)
    scored = [c["score"] for c in cards if c["score"] is not None]
    mean = sum(scored) / len(scored) if scored else 0
    median = statistics.median(scored) if scored else 0

    by_dom = {}
    for c in cards:
        by_dom.setdefault(c["domain"], []).append(c["score"])
    dom_rows = "".join(
        f"<tr><td>{k}</td><td>{len(v)}</td><td>{sum(x or 0 for x in v)/len(v):.4f}</td></tr>"
        for k, v in sorted(by_dom.items()))

    def stat(label, value):
        return f"<div class='stat'><div class='v'>{value}</div><div class='l'>{label}</div></div>"

    # Header title + subtitle derive from the merged trials, not a hardcoded count.
    n_bench = len(cards)
    trial_parts = [analysis.get("trial") or "main"]
    extra_trial = analysis.get("extra_trial")
    if extra_trial:
        trial_parts.append(f"新增 {extra_trial}")
    legacy_trial = analysis.get("legacy_trial")
    if legacy_trial:
        trial_parts.append(f"legacy {legacy_trial}")
    title = args.title or f"{n_bench} 个 Benchmark 评测报告 · Qwen3.6-35B-A3B"
    subtitle = (f"{' + '.join(trial_parts)} · 采样上限 500 题/benchmark · "
                f"完成率 @@TOTAL_C@@/@@TOTAL_A@@ = @@PCT@@%")

    cards_json = json.dumps(cards, ensure_ascii=False)

    DOM_OPTIONS = "".join(f"<option>{d}</option>" for d in dom_vals)
    SCR_OPTIONS = "".join(f"<option>{s}</option>" for s in scorer_vals)
    HEADER_STATS = "".join([
        stat("有产出 benchmark", f"{len(cards)}"),
        stat("有分数", f"{len(scored)}"),
        stat("完成率", f"{100*total_c/max(total_a,1):.2f}%"),
        stat("平均分（等权）", f"{mean:.4f}"),
        stat("中位分", f"{median:.4f}"),
        stat("满分 benchmark", f"{sum(1 for s in scored if s>=0.9999)}"),
    ])

    html_doc = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<style>
  :root { --bg:#0f1220; --card:#171b2e; --card2:#1d2238; --line:#2a3050;
           --fg:#e8eaf6; --mut:#9aa3c0; --acc:#6c8bff; --good:#3ddc97; --mid:#f5b041; --bad:#ff6b6b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 -apple-system,Segoe UI,Roboto,"PingFang SC","Microsoft YaHei",sans-serif; }
  header { padding:28px 34px 14px; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 4px; font-size:24px; }
  .sub { color:var(--mut); font-size:13px; }
  .stats { display:flex; flex-wrap:wrap; gap:10px; margin-top:18px; }
  .stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 18px; min-width:120px; }
  .stat .v { font-size:20px; font-weight:700; }
  .stat .l { color:var(--mut); font-size:12px; margin-top:2px; }
  .controls { position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:10px; align-items:center;
               padding:12px 34px; background:rgba(15,18,32,.96); backdrop-filter:blur(4px); border-bottom:1px solid var(--line); }
  .controls input, .controls select { background:var(--card); color:var(--fg); border:1px solid var(--line);
               border-radius:8px; padding:8px 12px; font-size:14px; }
  .controls input[type=text] { flex:1 1 240px; min-width:200px; }
  #count { color:var(--mut); font-size:13px; margin-left:auto; }
  main { padding:20px 34px 60px; }
  details.bench { background:var(--card); border:1px solid var(--line); border-radius:12px; margin-bottom:10px; overflow:hidden; }
  details.bench summary { list-style:none; cursor:pointer; padding:12px 16px; display:flex; align-items:center; gap:14px; }
  details.bench summary::-webkit-details-marker { display:none; }
  details.bench[open] summary { border-bottom:1px solid var(--line); background:var(--card2); }
  .score { font-size:15px; font-weight:700; padding:3px 10px; border-radius:7px; min-width:74px; text-align:center; }
  .s-high { background:#143d2f; color:var(--good); }
  .s-mid  { background:#3d3314; color:var(--mid); }
  .s-low  { background:#3d1a1a; color:var(--bad); }
  .s-none { background:#2a3050; color:var(--mut); }
  .bname { font-weight:700; font-size:15px; }
  .bid { color:var(--mut); font-size:12px; margin-top:1px; }
  .pill { font-size:11px; padding:2px 8px; border-radius:20px; background:#232949; color:#b6c0e8; border:1px solid var(--line); white-space:nowrap; }
  .comp { color:var(--mut); font-size:12px; text-align:right; margin-left:auto; white-space:nowrap; }
  .body { padding:18px 22px; }
  .meta-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:8px 24px; margin-bottom:16px;
               font-size:13px; color:var(--mut); border-bottom:1px solid var(--line); padding-bottom:14px; }
  .meta-grid b { color:var(--fg); font-weight:600; }
  .md h1 { font-size:20px; margin:6px 0 10px; }
  .md h2 { font-size:17px; margin:18px 0 8px; border-bottom:1px dashed var(--line); padding-bottom:4px; }
  .md h3 { font-size:15px; margin:14px 0 6px; }
  .md p { margin:8px 0; }
  .md ul, .md ol { margin:8px 0; padding-left:22px; }
  .md li { margin:3px 0; }
  .md table { border-collapse:collapse; margin:10px 0; width:100%; font-size:13px; }
  .md th, .md td { border:1px solid var(--line); padding:6px 10px; text-align:left; }
  .md th { background:var(--card2); }
  .md code { background:#0d1020; border:1px solid var(--line); padding:1px 5px; border-radius:4px; font-size:13px; }
  .md pre { background:#0d1020; border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; font-size:13px; }
  .md pre code { border:0; padding:0; }
  .md blockquote { margin:8px 0; padding:6px 14px; border-left:3px solid var(--acc); background:#151b31; border-radius:0 8px 8px 0; color:#c7cff0; }
  .md a { color:var(--acc); }
  .md hr { border:0; border-top:1px solid var(--line); margin:16px 0; }
  .warn { color:var(--bad); }
  .hidden { display:none !important; }
  .dom-summary table { border-collapse:collapse; font-size:13px; margin-top:8px; }
  .dom-summary th, .dom-summary td { border:1px solid var(--line); padding:5px 12px; text-align:left; }
  @media (max-width:700px) { main, .controls, header { padding-left:14px; padding-right:14px; } .comp { display:none; } }
</style>
</head>
<body>
<header>
  <h1>@@TITLE@@</h1>
  <div class="sub">@@SUBTITLE@@</div>
  <div class="stats">@@HEADER_STATS@@</div>
  <div class="dom-summary">
    <table>
      <thead><tr><th>领域</th><th>benchmark 数</th><th>等权平均分</th></tr></thead>
      <tbody>@@DOM_ROWS@@</tbody>
    </table>
  </div>
</header>
<div class="controls">
  <input type="text" id="q" placeholder="搜索名称 / id / 领域 / 打分器…" oninput="render()">
  <select id="dom" onchange="render()"><option value="">全部领域</option>@@DOM_OPTIONS@@</select>
  <select id="scr" onchange="render()"><option value="">全部打分器</option>@@SCR_OPTIONS@@</select>
  <select id="sort" onchange="render()">
    <option value="id">按 id</option><option value="score-desc">分数 ↓</option>
    <option value="score-asc">分数 ↑</option><option value="rows">题量 ↓</option><option value="pct">完成率 ↑</option>
  </select>
  <span id="count"></span>
</div>
<main id="list"></main>
<script>
const DATA = @@CARDS@@;
const $ = s => document.getElementById(s);
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function render(){
  const q = $('q').value.trim().toLowerCase();
  const dom = $('dom').value, scr = $('scr').value, sort = $('sort').value;
  let items = DATA.filter(c =>
    (!dom || c.domain===dom) && (!scr || c.scorer===scr) &&
    (!q || (c.id+' '+c.name+' '+c.domain+' '+c.scorer+' '+c.license).toLowerCase().includes(q)));
  if (sort==='score-desc') items.sort((a,b)=>(b.score??-1)-(a.score??-1));
  else if (sort==='score-asc') items.sort((a,b)=>(a.score??9e9)-(b.score??9e9));
  else if (sort==='rows') items.sort((a,b)=>(b.local_rows??0)-(a.local_rows??0));
  else if (sort==='pct') items.sort((a,b)=>(a.attempted? a.complete/a.attempted:1)-(b.attempted? b.complete/b.attempted:1));
  else items.sort((a,b)=>a.id<b.id?-1:a.id>b.id?1:0);
  $('count').textContent = items.length + ' / ' + DATA.length;
  $('list').innerHTML = items.map(c => `
    <details class="bench">
      <summary>
        <span class="score ${c.score_cls}">${c.score_txt}</span>
        <span><div class="bname">${esc(c.name)}</div><div class="bid">${esc(c.id)}</div></span>
        <span class="pill">${esc(c.domain)}</span><span class="pill">${esc(c.scorer)}</span>
        <span class="pill">${esc(c.license)}</span>
        <span class="comp">${c.complete}/${c.attempted} · ${c.pct}${c.local_rows!=null? ' · '+c.local_rows+'题':''}${c.failures>0? ' · <span class="warn">'+c.failures+' 空答判 0 分</span>':''}</span>
      </summary>
      <div class="body">
        <div class="meta-grid">
          <div><b>ID</b><br>${esc(c.id)}</div>
          <div><b>名称</b><br>${esc(c.name)}</div>
          <div><b>领域</b><br>${esc(c.domain)}</div>
          <div><b>打分器</b><br>${esc(c.scorer)}</div>
          <div><b>适配器</b><br>${esc(c.adapter)}</div>
          <div><b>许可</b><br>${esc(c.license)}</div>
          <div><b>本地题量</b><br>${c.local_rows!=null? c.local_rows:'—'}</div>
          <div><b>评测完成</b><br>${c.complete}/${c.attempted} · ${c.pct}</div>
          <div><b>空答判 0 分</b><br>${c.failures>0? c.failures:'0'}</div>
          <div><b>HF 来源</b><br>${esc(c.hf_path)}</div>
        </div>
        <div class="md">${c.md_html}</div>
      </div>
    </details>`).join('');
}
render();
</script>
</body>
</html>
"""
    html_doc = (html_doc
                .replace("@@CARDS@@", cards_json)
                .replace("@@DOM_OPTIONS@@", DOM_OPTIONS)
                .replace("@@SCR_OPTIONS@@", SCR_OPTIONS)
                .replace("@@HEADER_STATS@@", HEADER_STATS)
                .replace("@@DOM_ROWS@@", dom_rows)
                .replace("@@TITLE@@", title)
                .replace("@@SUBTITLE@@", subtitle)
                .replace("@@TOTAL_C@@", str(total_c))
                .replace("@@TOTAL_A@@", str(total_a))
                .replace("@@PCT@@", f"{100*total_c/max(total_a,1):.2f}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_doc, encoding="utf-8")
    print(f"[report] {len(cards)} benchmarks -> {args.out} ({args.out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

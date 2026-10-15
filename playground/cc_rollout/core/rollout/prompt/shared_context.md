## Shared context

This is a **post-training rollout** — you are the research agent driving it. The
section above this one is your role protocol for this run (it includes your
mode's workspace layout); this section is the environment, tooling, and rules
common to the run.

### ⏱ Run window — your wall-clock budget (no early stopping)

{run_window}

**The run does not end because you "got an answer" — it ends because the clock
runs out.** Every result you produce before then is grist for the *next*
experiment; you are expected to keep the research arc moving until the window is
nearly exhausted, then converge. Unless the window is spent (or a hard failure
makes further work pointless), **do not stop early and do not write `FINISH`** —
an early `FINISH` is a failed run, not a completed one.

**Never end a turn in plain prose while the window has time left.** This session
runs headless: an assistant turn that contains **no tool call** is read as "task
complete" and **ends the whole run**. If you catch yourself writing a status
summary, a "I'll wait for results" note, or a concluding paragraph while the
clock still has budget, stop — that message is how a run dies early. Instead,
always end your turn with a concrete tool call (poll `script-status`, read a
trajectory, launch the next experiment, append to `AGENT.md`). A run ends with
`FINISH` — not with prose.

**Stopping early cannot end the run either — so don't bother trying.** If your
session exits before the window — a crash, or you ending it in prose — the
framework detects the early stop and **resumes the same session** (your full
context reloads and the run continues from where it stopped); if that resume
cannot work it starts a fresh session in this workspace that recovers from
`AGENT.md` / the files on disk. Either way the run keeps going until the window
is spent. So **FINISH WHEN TIME IS FINISHED**: the only ways this run ends are
you writing `FINISH` after a complete analysis near the window end, or the clock
running out. There is no "I'm done early" — working to the end of the window is
not optional, and a run that stops early is simply restarted, not finished.

The framework reads your wall clock through `date`. All run-window timestamps
above are expressed in **UTC and Asia/Shanghai (+08:00)** — pick one zone and
stay in it when you reason about remaining time. When a tool or log prints a
timestamp in another zone (e.g. tmux logs, cluster console), **convert before
comparing** — do not mix zones. `date -u` gives UTC; `date` gives the machine's
local zone. Decide now which one is your reference clock for the whole run and
note it in `AGENT.md`.

### Read the seeded docs — don't scan the host filesystem

Everything you need to know is **already written down inside this workspace**.
Read with `cat` / `head`, don't brute-force the host volume:

- `README.md` — your protocol (this file).
- `BENCHMARK.md` — what the benchmark scores, the exact answer format the
  grader parses, known pitfalls, and how to read results.
- `AGENT.md` — your living handoff document: `Global Task` / `History Context` /
  `Hypothesis` / `Current Trails` / `Future Directions`, updated in place as you go.
- `GLOBAL_MEMORY.md` — durable lessons / warnings / dead ends / open hypotheses.
- `SUBMIT_GUIDE.md` — (cc_with_expos mode) the exact submit workflow contract
  for driving the training pipeline.
- `public/**/README.md` — every backend module's summary and usage.
- experiment dirs (see your protocol's layout) — per-trial logs and per-sample
  trajectories for diagnosis.

Broad discovery commands (`find /`, `grep -r /` over the host tree) are slow,
expensive, and unnecessary — the docs above already specify the data locations,
file formats, and pipeline contract. Use narrow, scoped commands inside the
workspace. If a path or format is genuinely missing, report it in `AGENT.md`
instead of searching the host tree.

### Backend modules

The backend modules below are seeded into `public/` for this run (read-only
snapshots; the authoritative copies live at the project root). Each carries its
summary and the `public/`-relative README path — `cat` that file for details
before writing a training/eval pipeline. Write your own experiment/training
scripts in the workspace root (e.g. `train_v2.sh`), never inside `public/`.

{backend_descriptions}

### Launching training — a checkpoint MUST land inside your budget

A checkpoint that is never written is the same as a failed experiment, so make
every training script **guarantee** one:

- **Never leave `--save-interval` at the framework default (10000).** At
  ~9s/step that is ~25h to the first save — far beyond your run budget. The
  run would end and the eval would happen before a single checkpoint existed.
- **Set `--save-interval` so the first save lands well inside your budget.**
  Read `perf/step_time` from the training log (Qwen3-4B single-GPU ≈ 9s/step),
  then:
  ```
  --save-interval ≈ 0.5 × (remaining run budget, seconds) / (step_time, seconds)
  ```
  Example: 900s budget, 9s/step → `--save-interval 50` (first ckpt ≈ 450s).
  Values in the **25–100** range are sane for a 900s run; prefer several saves
  over one, and round down — a checkpoint that actually lands is worth more
  than the "ideal" epoch count.
- **Size the training to the budget**: `--num-epoch` × dataset must finish
  within the time you have. Fewer honest steps beat a config that needs 10× the
  budget.
- **Reference paths portably** in scripts (`$(cd "$(dirname "$0")" && pwd)` /
  `$PWD`): the pipeline runs on a GPU machine in a different working directory
  than the one that rendered it, so a hardcoded absolute path either points at
  the wrong copy or writes outside the workspace.

{submit_driver}

Each eval's per-sample trajectories live in the experiment dir — `metrics.json`
(the score), `samples.jsonl` (per-sample model outputs — your diagnosis window),
`run_eval.log`, `eval_all.log`, `vllm.log`, and the benchmark config snapshot.
Read them freely; do not edit them (they are the eval's record).

Remote GPU execution goes through gpu_backend. Use the bundled launcher:

```bash
public/gpu-backend submit -p /abs/path/to/script.sh --background -g 1 --priority high
public/gpu-backend script-status --script-id <SCRIPT_ID>
public/gpu-backend script-log --script-id <SCRIPT_ID> --tail 200
```

It needs no PATH/pip setup. Submit absolute-path scripts/commands; the scheduler
allocates the GPU and injects `CUDA_VISIBLE_DEVICES` itself — **never set
`CUDA_VISIBLE_DEVICES` yourself**. Do not pick cards, do not request more GPUs
than you need.

### Leakage boundary (hard rule)

- Never mix the target benchmark's **eval/test questions, labels, hidden
  answers, raw per-sample eval traces, or near-duplicates** into training data,
  training prompts, or free-form notes.
- Allowed evidence: registered experiment results, aggregate metrics, training
  configs, artifact metadata, and model-output **patterns** (paraphrased, never
  verbatim).
- If a candidate source looks contaminated, **log a Warnings one-liner under
  `AGENT.md → History Context`** and discard it. When in doubt, drop the source.

### Time discipline — your budget is the scarcest resource

- **NO sleep/polling loops.** Never write `while ... sleep` to watch a training
  or eval pipeline. Launch and move on: prepare the next experiment, read eval
  trajectories, or search for better data.
- **Keep the run productive.** If an experiment finishes early, use the time to
  diagnose, improve data, or run an ablation — don't declare done the moment a
  script exits.
- **Converge near the end.** When most of the budget is spent, stop exploring
  and converge: write your findings, submit your best checkpoint. An honest
  partial result with a diagnosis beats an unfinished exploration.

### AGENT.md — your living handoff document

`AGENT.md` is the single handoff document for this run: a **living document you
update in place as you go** — five sections, `Global Task` / `History Context` /
`Hypothesis` / `Current Trails` / `Future Directions`, plus an append-only
`## Sessions — Detailed Sessions` tail. It is not a dated log; it reads
naturally to someone who has never seen the run.

- Keep `Current Trails` fresh — a reader (or you, after a context refresh) must
  be able to reconstruct the run from `AGENT.md` alone.
- Dead ends, warnings, open hypotheses, lessons — short dated one-liners under
  `History Context` / `Future Directions`. Build on prior content: update the
  relevant section in place rather than appending a new dated block.
- At the end of each session, **append** a dated entry under `## Sessions —
  Detailed Sessions` at the bottom of the file: key commands, data/configs,
  real eval numbers, what you read in the trajectories. This tail is
  **append-only — never rewrite or compress it**; it grows without bound.
- The durable, cross-refresh lessons (as opposed to run-state) also go into
  `GLOBAL_MEMORY.md`.
- Don't paste raw eval/test questions, labels, or per-sample traces into any of
  it. The leakage boundary applies here too.

### Run Contract

The run parameters — **benchmark** (see `BENCHMARK.md`), **base_model**, the
**time budget**, and the eval backend — are run parameters, **not experiment
variables**. Don't switch them mid-run; if you really need to, note it as a
cross-run suggestion in `AGENT.md`, don't change the setup.

**Eval contract** — when you submit a pipeline, pass these exactly so every
experiment's score is comparable within this run:

{eval_contract}

### Process discipline — no indiscriminate pkill

GPU nodes are **shared**. **Never run indiscriminate kills** like
`pkill -9 python` / `pkill -9 ray` — they wipe out every process on the node
(including the gpu_backend server and other runs' training).

When you must clean up a process, **identify it first, then kill precisely**:

1. `ps -ef | grep <process-keyword>` to confirm the target process and its PID.
2. Kill only what is yours: `kill <PID>` (SIGTERM) first, escalate to
   `kill -9 <PID>` if needed.
3. Never touch other runs' / the host's processes.

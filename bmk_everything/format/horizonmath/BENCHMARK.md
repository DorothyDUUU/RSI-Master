# HorizonMath

This integration pins the official HorizonMath repository at commit
`a34c47278697fda59d566ba11b2087da1bcf4454` and preserves its two-stage
generation/evaluation protocol. Model responses must define
`proposed_solution()`. The official evaluator executes the function and scores
numeric, best-known benchmark, and new-construction problems with the upstream
ground truths and validators.

The BmkEverything reward is exactly `official_summary.passed / total_problems`.
For best-known problems, a valid construction that only matches the baseline is
not counted as passed; it must beat the baseline. The complete official summary
and per-problem evaluation are retained alongside `reward.json`.

Splits: `default` (all 113), `calibration` (solvability 0), `numeric` (71),
`benchmark` (33), and `construction` (9).

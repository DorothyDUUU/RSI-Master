# Candidate release validation — 2026-10-16

The project integrates 24 benchmarks. See
[`benchmarks.json`](../bmk_everything/benchmarks.json) for the IDs,
[`datasets.json`](../bmk_everything/datasets.json) for data sources, and
[setup instructions](../bmk_everything/SETUP.md).

## Completed checks

- Selected framework suite: **88 passed, 1 skipped**, plus two passing subtests.
  The skipped module requires ARC-AGI, which is outside the supported catalogue.
- Unified dry-run checks: **24/24 passed** using an existing data installation
  and external checkouts, in an isolated checkout.
- Full local row mapping invariants checked for LEXam, MedXpertQA, WMDP-Bio,
  EconLogicQA, Earth Silver, CTI-MCQ and Pre-Flight.
- LegalBench uses the pinned DataPrep five-task, sample-weighted balanced accuracy
  implementation. LegalBench and CKQA gold-answer round trips scored 1.0.
- DataPrep source files were downloaded from revision
  `771688905d8c5c7e76acb83d6be615bb1027e066` and verified against the bootstrap
  script's SHA256 checksums. Materialized counts: LegalBench 413,
  MedCaseReasoning 897, CKQA 311.
- Original model/judge routing isolation and concurrent judging tests were
  included with the DataPrep adapter.
- The selected configurations were checked for legacy API credentials and
  machine-specific paths; actual credentials are not included.
- Both slime pipeline renderers generated a single-GPU candidate script that
  passed Bash syntax checking after the template escaping fix.

## Protocol notes

MedXpertQA preserves the original integration's documented **MM-data text-only**
variant: 2,000 MM-config questions and options, without sending images to the
text-only model. These scores are not interchangeable with a full multimodal
MedXpertQA evaluation.

CTI-MCQ retains the original excluded row. Of 2,499 remaining rows, four have
only A–C options. Their gold letters were checked against the original option
fields and did not shift after trailing blank-option filtering.

## Remaining verification

The following end-to-end checks remain:

- Not every dataset was downloaded into a fresh empty installation.
- The vLLM installation commands have not been verified in a new GPU environment.
- The full slime training → HF conversion → vLLM generation chain has not been
  rerun on the target GPU topology.
- Real candidate-model generation, judge calls and code-execution scoring have
  not been run for all 24 benchmarks.
- Pin judge models and evaluator parameters for comparable runs; changing
  graders changes the evaluation protocol.

Datasets, model weights, local environments and private experiment logs are not
included in the repository. Install each integration's required assets and services
before running an evaluation.

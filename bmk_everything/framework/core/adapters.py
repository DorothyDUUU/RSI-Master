from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    executable: bool
    description: str


ADAPTERS = {
    "legacy_runner": AdapterSpec("legacy_runner", True, "Per-benchmark run_benchmark.py"),
    "hf_exact_match": AdapterSpec("hf_exact_match", True, "Pinned Hugging Face deterministic exact match"),
    "hf_boolean": AdapterSpec("hf_boolean", True, "Pinned Hugging Face true/false exact match"),
    "hf_multiple_choice": AdapterSpec("hf_multiple_choice", True, "Pinned Hugging Face multiple choice"),
    "hf_numeric": AdapterSpec("hf_numeric", True, "Pinned Hugging Face numeric answer match"),
    "hf_json": AdapterSpec("hf_json", True, "Pinned Hugging Face canonical JSON match"),
    "bigbench": AdapterSpec("bigbench", True, "Pinned tasksource/BIG-bench task"),
    "inspect_eval": AdapterSpec("inspect_eval", False, "Pinned inspect_evals bridge; not enabled in the initial 200"),
    "code_execution": AdapterSpec("code_execution", False, "Requires a validated process sandbox"),
    "llm_judge": AdapterSpec("llm_judge", False, "Disallowed in the initial deterministic cohort"),
}


def get_adapter(name: str | None) -> AdapterSpec | None:
    return ADAPTERS.get(name or "")

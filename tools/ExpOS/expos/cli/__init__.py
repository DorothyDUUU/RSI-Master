"""Bash-callable CLI frontend for ExpOS.

入口: :func:`expos.cli.__main__.main` (等价于 ``python -m expos.cli``)

CLI 命名遵循 redesign.md::

    expos state add-new-worker --parent-worker-id worker_0001
    expos state get-current-worker
    expos state inspect-worker --worker-id worker_0001
    expos worker show-datapool --worker-id worker_0001
    expos worker inspect-data --data-id data_0001
    expos worker add-data --worker-id worker_0001 --artifact-path /path/train.jsonl
    expos worker update-readme --data-id data_0001 --summary "..."
    expos worker submit --worker-id worker_0001 --data-path /path/train.jsonl \
        --training-config /path/train_config.json --training-mode full_sft
    expos worker add-reviewer --worker-id worker_0001 \
        --experiment-ids exp_0001 exp_0002
    expos reviewer show-datapool --reviewer-id rev_0001
    expos reviewer inspect-data --data-id data_0001
    expos reviewer inspect-experiment --reviewer-id rev_0001 --experiment-id exp_0001
    expos reviewer submit-review --reviewer-id rev_0001 --experiment-id exp_0001 \
        --report /path/review.json

以及初始化 / 全局工具::

    expos init --root /path/to/experiment_os --task "..." --benchmark aime_2025
    expos events tail --limit 200
"""

from .main import main

__all__ = ["main"]

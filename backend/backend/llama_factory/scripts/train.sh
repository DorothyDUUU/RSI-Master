# Repo root derived from this script's location.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO/backend/backend/llama_factory/llama_factory"
source "$REPO/public/llama_factory_env/bin/activate"
which python
which llamafactory-cli 

llamafactory-cli train "$REPO/backend/backend/llama_factory/config/train_sft_gsm8k.yaml"
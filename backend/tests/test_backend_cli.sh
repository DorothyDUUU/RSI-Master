# test basic usage

# Repo root is derived from this script's location (test/ lives under <repo>/backend/).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/backend"
python gpu_backend/main.py exec -c "nvidia-smi"

python gpu_backend/main.py submit -p "$REPO/backend/backend/vllm/scripts/run_qwen_3-1.8.sh --background -t 0



python gpu_backend/main.py submit -p "$REPO/backend/backend/llama_factory/scripts/train.sh --background -t 3600


python gpu_backend/main.py submit -p "$REPO/backend/backend/ptb_gsm8k/eval.sh --background -t 3600

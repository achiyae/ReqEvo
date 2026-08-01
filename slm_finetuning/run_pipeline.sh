#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use a persistent cache directory that is safe on SLURM clusters.
# You can override it by exporting HF_CACHE_DIR=/path/to/shared/cache
# before running this script, or by setting HF_HOME directly.
CACHE_ROOT="${HF_CACHE_DIR:-${HF_HOME:-$SCRIPT_DIR/.cache}}"
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE TORCH_HOME
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

VENV_DIR="$SCRIPT_DIR/venv"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  source "$VENV_DIR/Scripts/activate"
else
  echo "Virtual environment not found. Run setup_env.sh first." >&2
  exit 1
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON_CMD="$VENV_DIR/bin/python"
elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
  PYTHON_CMD="$VENV_DIR/Scripts/python.exe"
else
  echo "Python executable not found in virtual environment." >&2
  exit 1
fi

echo "Starting SLM Fine-Tuning Pipeline..."

echo ""
echo "--- Training Phi-3.5-mini-instruct ---"
"$PYTHON_CMD" train.py --model_id "microsoft/Phi-3.5-mini-instruct" --epochs 4

echo ""
echo "--- Training Qwen2.5-7B-Instruct ---"
"$PYTHON_CMD" train.py --model_id "Qwen/Qwen2.5-7B-Instruct" --epochs 3

echo ""
echo "--- Evaluating Phi-3.5-mini-instruct ---"
"$PYTHON_CMD" evaluate.py --model_id "microsoft/Phi-3.5-mini-instruct" --adapter_dir "./lora_adapter_Phi-3.5-mini-instruct" --test_file "test_set_Phi-3.5-mini-instruct.json"

echo ""
echo "--- Evaluating Qwen2.5-7B-Instruct ---"
"$PYTHON_CMD" evaluate.py --model_id "Qwen/Qwen2.5-7B-Instruct" --adapter_dir "./lora_adapter_Qwen2.5-7B-Instruct" --test_file "test_set_Qwen2.5-7B-Instruct.json"

echo ""
echo "========================================================"
echo "Pipeline completed! Review the accuracy percentages above."
echo "To patch your JSON files with predictions and certainty scores, pick the winning model and run one of these commands:"
echo "For Qwen2.5:"
echo "  python predict.py --model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir ./lora_adapter_Qwen2.5-7B-Instruct"
echo "For Phi-3.5:"
echo "  python predict.py --model_id microsoft/Phi-3.5-mini-instruct --adapter_dir ./lora_adapter_Phi-3.5-mini-instruct"

#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Python virtual environment..."

python_cmd=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        python_cmd="$candidate"
        break
    fi
done

if [[ -z "$python_cmd" ]]; then
    echo "Could not find Python 3. Install Python 3 or add it to PATH." >&2
    exit 1
fi

echo "Using Python interpreter: $(command -v "$python_cmd")"

python_bin="$python_cmd"

# Create venv
"$python_bin" -m venv venv

# Activate venv
venv_path="$(cd "$(dirname "$0")" && pwd)/venv"
if [[ -f "$venv_path/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$venv_path/bin/activate"
else
    echo "Failed to create virtual environment." >&2
    exit 1
fi

venv_python="$venv_path/bin/python"

echo "Upgrading pip..."
"$venv_python" -m pip install --upgrade pip

echo "Installing required AI libraries..."
"$venv_python" -m pip install torch transformers peft trl datasets bitsandbytes accelerate

echo
echo "========================================================"
echo "Environment setup complete!"
echo "The virtual environment 'venv' has been created and populated."
echo "To activate it manually in the future, run:"
echo "  . ./venv/bin/activate"

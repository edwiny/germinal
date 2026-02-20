#!/usr/bin/env bash
# Purpose: Create .venv, install dependencies, and initialise the database.
# Run from the orchestrator/ directory: bash scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Creating virtual environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists, skipping creation"
fi

# Activate for the rest of this script.
# shellcheck disable=SC1091
source .venv/bin/activate

echo ""
echo "=== Installing Python dependencies ==="
pip install --quiet --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Initialising database ==="
python - <<'EOF'
import yaml
from storage.db import init_db

with open("config.yaml") as f:
    config = yaml.safe_load(f)

db_path = config["paths"]["db"]
init_db(db_path)
print(f"Database initialised at: {db_path}")
EOF

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: source .venv/bin/activate"
echo "Then run: python main.py"

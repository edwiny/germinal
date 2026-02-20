#!/usr/bin/env bash
# Purpose: Activate .venv and run the full test suite with verbose output.
# Run from the orchestrator/ directory: bash scripts/run_tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

pytest tests/ -v "$@"

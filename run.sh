#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
cd "$TASK_DIR"
exec python -u "$TASK_DIR/smoke_test_lora.py" "$@"

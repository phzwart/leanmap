#!/usr/bin/env bash
# Thin wrapper — prefer run_phase1.py (macOS bash 3.2-safe).
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python "${ROOT}/examples/exploratory/run_phase1.py" "$@"

#!/usr/bin/env bash
# Thin wrapper around benchmark.py for manual smokes.
#
# Usage:
#   scripts/run_eval.sh <agent_import> [dev_size] [concurrency] [extra benchmark flags...]
#
# Examples:
#   MH_N_TASKS=3 scripts/run_eval.sh agents.baseline_single
#   scripts/run_eval.sh agents.baseline_react 30 16 --run-dir logs/smoke
#
# Sources .env from this experiment dir (QNAIGC_API_KEY). Run inside a Slurm job.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"
if [ -f .env ]; then set -a; . ./.env; set +a; fi

AGENT="${1:?usage: run_eval.sh <agent_import> [dev_size] [concurrency] [flags...]}"
shift || true

ARGS=(--agent "$AGENT")
if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then ARGS+=(--dev-size "$1"); shift; fi
if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then ARGS+=(--concurrency "$1"); shift; fi

exec uv run python benchmark.py "${ARGS[@]}" "$@"

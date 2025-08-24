#!/usr/bin/env bash
set -euo pipefail

# Small launcher for transduction jobs using uv + torchrun.
# Usage:
#   ./scripts/uv_torchrun_transduction.sh <data-gen|sft|rl> [--nproc N] [-- ...args]
# Examples:
#   ./scripts/uv_torchrun_transduction.sh data-gen -- --data_dir training_data --output transduction/train_dataset.json
#   ./scripts/uv_torchrun_transduction.sh sft --nproc 2
#   ./scripts/uv_torchrun_transduction.sh rl --nproc 2

usage() {
  cat <<'USAGE'
Usage:
  uv_torchrun_transduction.sh <data-gen|sft|rl> [--nproc N] [-- ...args]

Options:
  --nproc N   Number of processes per node (default: 1)
  --          Pass remaining args to the underlying module (primarily for data-gen)

Examples:
  ./scripts/uv_torchrun_transduction.sh data-gen -- --data_dir training_data --output transduction/train_dataset.json
  ./scripts/uv_torchrun_transduction.sh sft --nproc 2
  ./scripts/uv_torchrun_transduction.sh rl --nproc 2
USAGE
}

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed or not on PATH. See https://docs.astral.sh/uv/" >&2
  exit 1
fi

# Move to repo root so uv picks up pyproject.toml
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

TASK="$1"; shift || true
NPROC=1

FORWARD_ARGS=()
while (( "$#" )); do
  case "$1" in
    --nproc)
      shift
      if [[ -z "${1:-}" ]]; then echo "--nproc requires a value" >&2; exit 2; fi
      NPROC="$1"; shift
      ;;
    --)
      shift
      # Everything after -- is forwarded
      FORWARD_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

MODULE=""
case "$TASK" in
  data-gen)
    MODULE="transduction.data_gen"
    ;;
  sft)
    MODULE="transduction.training.sft"
    ;;
  rl)
    MODULE="transduction.training.rl"
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    usage
    exit 2
    ;;
esac

set -x
uv run torchrun --standalone --nproc_per_node="${NPROC}" -m "${MODULE}" "${FORWARD_ARGS[@]}"


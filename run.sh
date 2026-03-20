#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: bash run.sh <questions_txt_path> <predictions_out_path>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/hybrid_rag.py" \
  --questions-file "$1" \
  --answers-file "$2" \
  --rrf-k 30 \
  --bm25-weight 1.5 \
  --dense-weight 0.5 \
  --candidate-multiplier 12 \
  --top-k 8

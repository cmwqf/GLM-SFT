#!/usr/bin/env bash
# Deploy the DCP sharded-load patch into an installed verl.
#
# Usage:
#   ./apply_patch.sh [VERL_ENGINE_FSDP_DIR]
# Default target dir:
#   /opt/pip-packages/lib/python3.11/site-packages/verl/workers/engine/fsdp
#
# Backs up the original transformer_impl.py to transformer_impl.py.bak-<ts> (once),
# then copies the patched file in. Idempotent-ish: re-running re-copies the patched file.
set -euo pipefail

TARGET_DIR="${1:-/opt/pip-packages/lib/python3.11/site-packages/verl/workers/engine/fsdp}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DST="$TARGET_DIR/transformer_impl.py"
SRC="$HERE/transformer_impl.py"

if [[ ! -f "$DST" ]]; then
  echo "ERROR: $DST not found. Is verl installed at $TARGET_DIR ?" >&2
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
if ! ls "$TARGET_DIR"/transformer_impl.py.bak-* >/dev/null 2>&1; then
  cp -v "$DST" "$TARGET_DIR/transformer_impl.py.bak-$TS"
else
  echo "backup already exists, skipping backup"
fi

cp -v "$SRC" "$DST"
python3.11 -m py_compile "$DST" && echo "OK: patched + compiles"
echo
echo "Enable at training time by exporting:"
echo "  export VERL_DCP_CKPT_PATH=/data/models/GLM-4.5-Air-dcp"

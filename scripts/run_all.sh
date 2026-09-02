#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR=${1:-build}
for exe in "$BUILD_DIR"/[0-9][0-9]_*; do
  if [[ -x "$exe" ]]; then
    echo "==== $(basename "$exe") ===="
    "$exe"
  fi
done

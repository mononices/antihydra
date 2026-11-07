#!/usr/bin/env bash
# Headless Ghidra decompilation of every function in a binary.
# Usage: ghidra-decompile <binary> [output.c]
set -euo pipefail

BIN="${1:?usage: ghidra-decompile <binary> [output.c]}"
OUT="${2:-/tmp/decompiled.c}"
PROJECT_DIR="${TMPDIR:-/tmp}/ghidra-project"

rm -rf "$PROJECT_DIR"
/opt/ghidra/support/analyzeHeadless "$PROJECT_DIR" antihydra \
  -import "$(realpath "$BIN")" \
  -scriptPath /opt/ghidra/scripts \
  -postScript DecompileAll.java "$OUT" \
  -deleteProject >/dev/null

echo "decompiled output: $OUT"

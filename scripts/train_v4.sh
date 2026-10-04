#!/bin/bash
# Lanceur entraînement v4 figé (remplace /tmp/opencode/gcntrain.sh, éphémère).
# Force le code source du repo (src 4.0.0), jamais le pip 2.5.0 divergent.
# Usage: scripts/train_v4.sh --data-dir ... --val-dir ... [options gcn-train]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/gcn-python/src"
VER="$(python3 -c 'import gcn_python; print(gcn_python.__version__)')"
if [ "$VER" != "4.0.0" ]; then
  echo "REFUS: gcn_python==$VER (attendu 4.0.0, src repo)" >&2
  exit 1
fi
echo "gcn_python src==$VER OK"
exec python3 -c 'import sys; from gcn_python.training.train import train_cmd; train_cmd.main(sys.argv[1:])' "$@"

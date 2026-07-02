#!/usr/bin/env bash
# End-to-end pipeline for judges. One command from docker-compose runs this.
#
#   1. Precompute the FAISS index (only if not already cached in the volume)
#   2. Rank deterministically -> submission.csv
#   3. Validate the CSV against the official rules
set -euo pipefail

CANDIDATES="${CANDIDATES_PATH:-/data/candidates.jsonl}"
ARTIFACTS="${ARTIFACTS_DIR:-/app/artifacts}"
OUT="${OUT_PATH:-/out/submission.csv}"

mkdir -p "$ARTIFACTS" "$(dirname "$OUT")"

if [[ ! -f "$CANDIDATES" ]]; then
  echo "ERROR: candidates file not found at $CANDIDATES" >&2
  echo "Mount it via the ./data volume (see docker-compose.yml)." >&2
  exit 1
fi

if [[ -f "$ARTIFACTS/faiss.index" && -f "$ARTIFACTS/candidate_ids.txt" ]]; then
  echo ">> Reusing cached FAISS index in $ARTIFACTS (delete it to rebuild)."
else
  echo ">> [1/3] Precomputing embeddings + FAISS index (no time limit)..."
  python precompute.py --candidates "$CANDIDATES" --out-dir "$ARTIFACTS"
fi

echo ">> [2/3] Ranking (deterministic, CPU-only, offline)..."
python rank.py --candidates "$CANDIDATES" --out "$OUT"

echo ">> [3/3] Validating submission..."
python validate_submission.py "$OUT" || {
  echo "Validation reported issues (see above)." >&2
  exit 1
}

echo ">> Done. Submission written to $OUT"

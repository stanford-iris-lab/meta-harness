#!/usr/bin/env bash
# Download + extract a SpreadsheetBench dataset into this data/ directory.
#
# Data provenance: https://github.com/RUCKBReasoning/SpreadsheetBench (git-LFS blobs
# under data/). We pull the real blob via the media.githubusercontent.com LFS endpoint.
#
# Usage:
#   bash data/fetch_data.sh                       # sample_data_200 (default)
#   bash data/fetch_data.sh spreadsheetbench_912_v0.1
#   bash data/fetch_data.sh spreadsheetbench_verified_400
#
# Run this in a Slurm job or salloc, NOT on the login node if it is large.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${1:-sample_data_200}"
REPO="RUCKBReasoning/SpreadsheetBench"
RAW_URL="https://raw.githubusercontent.com/${REPO}/main/data/${SOURCE}.tar.gz"
LFS_URL="https://media.githubusercontent.com/media/${REPO}/main/data/${SOURCE}.tar.gz"
TARBALL="${DATA_DIR}/${SOURCE}.tar.gz"
DEST="${DATA_DIR}/${SOURCE}"

if ! gzip -t "$TARBALL" 2>/dev/null; then
  echo "Downloading ${RAW_URL}"
  curl -fL --retry 3 "$RAW_URL" -o "$TARBALL"
  # Larger sets may be git-LFS: raw then returns a text pointer, not gzip. Refetch.
  if ! gzip -t "$TARBALL" 2>/dev/null; then
    echo "raw was not gzip (LFS pointer?); retrying ${LFS_URL}"
    curl -fL --retry 3 "$LFS_URL" -o "$TARBALL"
  fi
fi

echo "Extracting ${TARBALL} -> ${DEST}/"
rm -rf "$DEST"
mkdir -p "$DEST"
tar -xzf "$TARBALL" -C "$DEST"

# Normalize a single wrapping directory (…/<SOURCE>/<SOURCE>/dataset.json -> …/<SOURCE>/).
inner=$(find "$DEST" -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -n "$inner" ] && [ ! -e "$DEST/dataset.json" ] && [ ! "$(find "$DEST" -maxdepth 1 -name '*.jsonl' | head -1)" ]; then
  if [ -e "$inner/dataset.json" ] || [ -n "$(find "$inner" -maxdepth 1 -name '*.jsonl' | head -1)" ]; then
    shopt -s dotglob
    mv "$inner"/* "$DEST"/ && rmdir "$inner" || true
    shopt -u dotglob
  fi
fi

echo "Done. Verify with:"
echo "  cd $(dirname "$DATA_DIR") && uv run python -m data.loader --source ${SOURCE} --list | head"

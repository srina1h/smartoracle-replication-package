#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "$0")" && pwd)"
cache_dir="$(mktemp -d "${TMPDIR:-/tmp}/smartoracle-mpl.XXXXXX")"
trap 'rm -rf "$cache_dir"' EXIT

export MPLBACKEND=Agg
export MPLCONFIGDIR="$cache_dir"
export LOKY_MAX_CPU_COUNT=1

cd "$package_dir"
./validate_package.sh

echo "Running root-cause propagation analysis"
(cd analysis/clustering && python3 analyze_root_cause_propagation.py)

echo "Running reasoning-trace analysis"
(cd analysis/reasoning && python3 main.py)

echo "Reproduction complete"

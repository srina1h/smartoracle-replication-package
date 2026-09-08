#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$package_dir"

python3 - <<'PY'
import json
from pathlib import Path

root = Path.cwd()

labels = json.loads((root / "data/labels/manual_labels_combined.json").read_text())
assert len(labels) == 710, f"expected 710 labels, found {len(labels)}"

sessions = list((root / "analysis/reasoning/logs/sessions").glob("*.json"))
assert len(sessions) == 234, f"expected 234 sessions, found {len(sessions)}"

features = json.loads((root / "data/feature_vectors.json").read_text())
feature_vectors = features.get("feature_vectors", features)
assert len(feature_vectors) == 710, f"expected 710 feature vectors, found {len(feature_vectors)}"

propagation = json.loads((root / "analysis/clustering/results/root_cause_propagation_analysis.json").read_text())
summary = propagation["summary"]
assert round(summary["baseline_avg_accuracy"] * 100, 2) == 77.34
assert round(summary["kmeans_avg_accuracy"] * 100, 2) == 89.15
assert summary["num_patterns"] == 14
assert sum(item["optimal_k"] for item in propagation["kmeans_results"].values()) == 94

smart = json.loads((root / "baseline/results/smartoracle_batch_summary.json").read_text())
baseline = json.loads((root / "baseline/results/sequential_baseline_batch_summary.json").read_text())
assert smart["count"] == 44 and round(smart["elapsed_s"], 2) == 901.30
assert baseline["count"] == 45 and round(baseline["elapsed_s"], 2) == 4084.16

stats = json.loads((root / "analysis/reasoning/results/basic_stats.json").read_text())
assert stats["total_sessions"] == 234
assert stats["decision_distribution"] == {"SKIP": 107, "REPORT": 114, "UNKNOWN": 13}

print("PASS: 710 labels and feature vectors, 234 traces, 94 medoids, and retained summaries verified")
PY

if find . -path './.git' -prune -o -type f -name '.env' -print -quit | grep -q .; then
  echo "FAIL: package contains a .env file" >&2
  exit 1
fi

if grep -RIlE '/Users/|AIza[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z]{20,}' . \
  --exclude-dir='.git' --exclude='validate_package.sh' | grep -q .; then
  echo "FAIL: package contains an absolute user path or credential-like string" >&2
  exit 1
fi

echo "PASS: no .env files, absolute user paths, or common credential patterns found"

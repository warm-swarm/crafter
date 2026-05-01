#!/usr/bin/env bash
# Run all sweep-analysis scripts on a baseline/treatment pair.
# Outputs land in <output_dir>/.
#
# Usage:
#   ./run_all.sh <baseline.jsonl> <treatment.jsonl> <output_dir> [<random.jsonl>]
#
# When <random.jsonl> is supplied, condition=random rows are added to
# pool_aggregates.csv and a random-policy floor line is drawn on b_vs_t.pdf.
#
# Override the Python interpreter via PYTHON env var, e.g.
#   PYTHON="uv run python" ./run_all.sh base.jsonl treat.jsonl out/

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <baseline.jsonl> <treatment.jsonl> <output_dir> [<random.jsonl>]" >&2
  exit 2
fi

baseline=$(realpath "$1")
treatment=$(realpath "$2")
output_dir=$3
random_path=""
if [[ $# -eq 4 ]]; then
  random_path=$(realpath "$4")
  [[ -f "$random_path" ]] || { echo "random not found: $random_path" >&2; exit 1; }
fi

[[ -f "$baseline" ]] || { echo "baseline not found: $baseline" >&2; exit 1; }
[[ -f "$treatment" ]] || { echo "treatment not found: $treatment" >&2; exit 1; }

mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")

PYTHON=${PYTHON:-python}
analysis_dir=$(dirname "$(realpath "$0")")
cd "$analysis_dir"

if [[ -n "$random_path" ]]; then
  random_arg=(--random "$random_path")
  echo ">> random-floor enabled ($random_path)"
else
  random_arg=()
  echo ">> random-floor skipped (no random.jsonl supplied)"
fi

echo ">> heatmap (baseline)"
$PYTHON plot_heatmap.py --input "$baseline" \
    --output "$output_dir/baseline_heatmap.pdf" --title baseline

echo ">> heatmap (treatment)"
$PYTHON plot_heatmap.py --input "$treatment" \
    --output "$output_dir/treatment_heatmap.pdf" --title treatment

echo ">> pool aggregates"
$PYTHON compute_pool_aggregates.py --baseline "$baseline" \
    --treatment "$treatment" "${random_arg[@]}" \
    --output "$output_dir/pool_aggregates.csv"

echo ">> baseline-vs-treatment plot"
$PYTHON plot_b_vs_t.py --baseline "$baseline" --treatment "$treatment" \
    "${random_arg[@]}" --output "$output_dir/b_vs_t.pdf"

echo ">> achievement breakdown"
$PYTHON achievement_breakdown.py --baseline "$baseline" \
    --treatment "$treatment" --output "$output_dir/achievement_breakdown.csv"

echo ">> achievement breakdown plots"
$PYTHON plot_achievement_breakdown.py --baseline "$baseline" \
    --treatment "$treatment" \
    --output-prefix "$output_dir/achievement_breakdown"

echo ">> length-vs-return plots"
$PYTHON plot_length_vs_return.py --baseline "$baseline" \
    --treatment "$treatment" \
    --output-prefix "$output_dir/length_vs_return"

for pool in train test; do
  echo ">> paired Wilcoxon ($pool pool)"
  $PYTHON paired_test.py --baseline "$baseline" --treatment "$treatment" \
      --pool "$pool" --output "$output_dir/paired_test_${pool}.txt"
done

echo
echo "all analysis written to $output_dir:"
ls -1 "$output_dir"

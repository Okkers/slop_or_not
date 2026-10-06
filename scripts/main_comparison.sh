#!/usr/bin/env bash
# The whole flow for ONE dataset (one CSV per LLM), see README section 0:
#   1. add the COOC column (skipped if the CSV already has one)
#   2. train + test GCN, GAT, SAGE and MLP on the WAN graphs, then on the COOC graphs (same split)
#   3. SVM baseline (on the WANs)
#   4. print the results table for this dataset
#
# Usage:   bash scripts/main_comparison.sh data/qwen2.5-1.5b.csv
# Options: ITR=3 bash scripts/main_comparison.sh ...                       seeds per model (default 5)
#          ID=E1 bash scripts/main_comparison.sh ...                       tag for results.csv (default main)
#          EXTRA="--train_epochs 3" bash scripts/main_comparison.sh ...    extra flags for main.py
set -e
DATA=${1:?"give the dataset CSV, e.g.  bash scripts/main_comparison.sh data/qwen2.5-1.5b.csv"}
ID=${ID:-main}
ITR=${ITR:-5}
EXTRA=${EXTRA:-}

echo "== 1/4 co-occurrence column"
if head -1 "$DATA" | grep -q "COOC"; then
  echo "$DATA already has a COOC column"
else
  uv run python -m data_factory.cooc --data_path "$DATA"
fi

echo "== 2/4 GNNs on the WAN graphs, then on the COOC graphs"
for GRAPH in wan cooc; do
  for MODEL in GCN GAT SAGE MLP; do
    echo "-- $MODEL on $GRAPH"
    uv run main.py --data_path "$DATA" --graph_type $GRAPH --model $MODEL --itr $ITR --model_id $ID --verbose false $EXTRA
  done
done

echo "== 3/4 SVM baseline"
uv run run_baselines.py --data_path "$DATA" --model_id $ID

echo "== 4/4 results for $(basename "$DATA")"
uv run python -m vis.summarize --model_id $ID --data "$(basename "$DATA")"
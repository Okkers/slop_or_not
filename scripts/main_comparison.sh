#!/usr/bin/env bash
# E3 + required baselines: every GNN (5 seeds) and the SVM / WL-kernel / RelEnt baselines,
# all on the same split. Usage:  bash scripts/main_comparison.sh data/persuade_w_wan.csv
set -e
DATA=${1:-data/persuade_w_wan.csv}
ID=${ID:-E3}
for MODEL in GCN GAT GIN SAGE MLP; do
  uv run main.py --data_path "$DATA" --model $MODEL --itr 5 --model_id $ID --verbose false
done
uv run run_baselines.py --data_path "$DATA" --model_id $ID
uv run python -m vis.summarize --model_id $ID

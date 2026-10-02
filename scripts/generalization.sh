#!/usr/bin/env bash
# Robustness: (a) unseen essay prompts (topics) at test time, (b) E2: unseen LLM at test time.
# Usage:  bash scripts/generalization.sh data/persuade_w_wan.csv "Qwen/Qwen2.5-1.5B-Instruct"
# (b) needs a 'generator' column in the CSV (see README).
set -e
DATA=${1:-data/persuade_w_wan.csv}
TEST_GEN=${2:-}
for MODEL in GCN GAT GIN SAGE; do
  uv run main.py --data_path "$DATA" --model $MODEL --itr 3 --verbose false --model_id cross_topic --split group --group_col task_prompt
done
uv run run_baselines.py --data_path "$DATA" --model_id cross_topic --split group --group_col task_prompt
uv run python -m vis.summarize --model_id cross_topic

if [ -n "$TEST_GEN" ]; then
  for MODEL in GCN GAT GIN SAGE; do
    uv run main.py --data_path "$DATA" --model $MODEL --itr 3 --verbose false --model_id E2 --split generator --test_generators "$TEST_GEN"
  done
  uv run run_baselines.py --data_path "$DATA" --model_id E2 --split generator --test_generators "$TEST_GEN"
  uv run python -m vis.summarize --model_id E2
fi

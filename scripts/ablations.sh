#!/usr/bin/env bash
# Feature + edge-weight ablations and hyper-parameter sensitivity (proposal, "Ablation studies").
# Usage:  bash scripts/ablations.sh data/persuade_w_wan.csv [GCN]
set -e
DATA=${1:-data/persuade_w_wan.csv}
MODEL=${2:-GCN}
RUN="uv run main.py --data_path $DATA --model $MODEL --itr 3 --verbose false"

# node features (add perplexity / embeddings later with --extra_node_cols)
for NF in "count,freq,onehot" "freq,onehot" "count,freq" "onehot" ""; do
  $RUN --model_id abl_features --node_features "$NF"
done
# edges: weighted vs unweighted, directed vs undirected
$RUN --model_id abl_edges --edge_weight wan
$RUN --model_id abl_edges --edge_weight binary
$RUN --model_id abl_edges --undirected true
# sensitivity: layers, hidden size, learning rate, WAN window width
for L in 1 2 3 4;            do $RUN --model_id sens_layers --num_layers $L; done
for H in 16 32 64 128;       do $RUN --model_id sens_hidden --hidden_dim $H; done
for LR in 1e-4 3e-4 1e-3 3e-3; do $RUN --model_id sens_lr --learning_rate $LR; done
for W in 2 3 5 8;            do $RUN --model_id sens_window --window $W; done

for ID in abl_features abl_edges; do uv run python -m vis.summarize --model_id $ID --by model_id,model,node_features,edge_weight,undirected; done
uv run python -m vis.summarize --model_id sens_layers --by model,num_layers
uv run python -m vis.summarize --model_id sens_hidden --by model,hidden_dim
uv run python -m vis.summarize --model_id sens_lr --by model,learning_rate
uv run python -m vis.summarize --model_id sens_window --by model,window

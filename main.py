"""
Train / evaluate graph classifiers (human = 0 vs machine = 1) on function-word WANs
or on all-word co-occurrence graphs.

Quick start (see README.md for more):
    uv run main.py --data_path data/persuade_w_wan.csv --model GCN
    uv run main.py --data_path data/persuade_w_wan.csv --model GAT --itr 5
    uv run main.py --data_path data/persuade_w_wan.csv --model GCN --graph_type cooc   # needs the COOC column
"""
import argparse
import json
import os

import numpy as np

from data_factory.splits import split_file_path
from exp.exp_main import Exp_Main
from exp.tools import set_seed


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def get_parser():
    p = argparse.ArgumentParser(description="Slop or not? GNN training on WAN or co-occurrence graphs")

    g = p.add_argument_group("basic")
    g.add_argument("--is_training", type=int, default=1, help="1 = train + test, 0 = only test a saved checkpoint")
    g.add_argument("--model", default="GCN", help="GCN | GAT | SAGE | MLP (no-graph control)")
    g.add_argument("--model_id", default="run", help="free-text tag for this experiment, e.g. E3 or abl_edges")
    g.add_argument("--itr", type=int, default=1, help="number of repeats with different seeds (seed, seed+1, ...)")
    g.add_argument("--seed", type=int, default=2026, help="training seed (init, batch order)")
    g.add_argument("--verbose", type=str2bool, default=True)
    g.add_argument("--log_every", type=int, default=10, help="print every N epochs")

    g = p.add_argument_group("data")
    g.add_argument("--data_path", default="data/persuade_w_wan.csv", help="any CSV from data_factory/data_generator.py;")
    g.add_argument("--function_words", default="data/function_words.txt")
    g.add_argument("--graph_type", default="wan", choices=["wan", "cooc"],
                   help="wan: 277 function-word nodes; cooc: every word of the text is a node, read from the "
                        "COOC column (add it with: uv run python -m data_factory.cooc --data_path ...)")
    g.add_argument("--wan_source", default="text", choices=["text", "csv"],
                   help="wan only. text: rebuild the WAN from output_text (needed to vary --window / function "
                        "words); csv: use the precomputed WAN column")
    g.add_argument("--window", type=int, default=5,
                   help="window width. wan: used to build the WAN; cooc: must match the window cooc.py used")
    g.add_argument("--node_features", default="count,freq,emb",
                   help="comma list from: count, freq, emb (pre-trained word vectors), onehot (wan only)")
    g.add_argument("--embeddings", default="glove-wiki-gigaword-100",
                   help="pre-trained word vectors for 'emb' (any gensim-downloader name, "
                        "e.g. word2vec-google-news-300)")
    g.add_argument("--extra_node_cols", default="",
                   help="CSV columns with a JSON list of one value per function word (e.g. perplexity)")
    g.add_argument("--edge_weight", default="wan", choices=["wan", "binary"])
    g.add_argument("--undirected", type=str2bool, default=False, help="symmetrise the WAN: (W + W^T)/2")
    g.add_argument("--cache_dir", default="data/cache")
    g.add_argument("--use_cache", type=str2bool, default=True)

    g = p.add_argument_group("split (identical for every model; saved in --split_dir)")
    g.add_argument("--split", default="random", choices=["random", "group", "generator"])
    g.add_argument("--split_seed", type=int, default=2026)
    g.add_argument("--val_ratio", type=float, default=0.15)
    g.add_argument("--test_ratio", type=float, default=0.15)
    g.add_argument("--group_col", default="task_prompt", help="--split group: column whose values are held out")
    g.add_argument("--test_groups", default="", help="--split group: values to hold out, separated by |")
    g.add_argument("--gen_col", default="generator", help="--split generator: column naming the LLM ('human' for humans)")
    g.add_argument("--test_generators", default="", help="--split generator: LLM(s) only seen at test time, separated by |")
    g.add_argument("--split_dir", default="splits", help="commit this folder so the whole team uses the same splits")
    g.add_argument("--split_file", default="", help="use this exact split JSON instead")

    g = p.add_argument_group("model")
    g.add_argument("--num_layers", type=int, default=2)
    g.add_argument("--hidden_dim", type=int, default=64)
    g.add_argument("--dropout", type=float, default=0.3)
    g.add_argument("--readout", default="sum", choices=["mean", "sum", "max", "meanmax"])
    g.add_argument("--norm", default="batch", choices=["batch", "none"])
    g.add_argument("--root_weight", type=str2bool, default=True,
                   help="GCN/GAT: add a separate linear term for the node's own features (skip connection)")
    g.add_argument("--heads", type=int, default=4, help="GAT attention heads")
    g.add_argument("--attn_dropout", type=float, default=0.1, help="GAT attention dropout")
    g.add_argument("--gat_v2", type=str2bool, default=False)

    g = p.add_argument_group("optimisation")
    g.add_argument("--train_epochs", type=int, default=200)
    g.add_argument("--patience", type=int, default=25, help="epochs without validation improvement before stopping")
    g.add_argument("--es_metric", default="loss", choices=["loss", "f1", "auc", "accuracy"],
                   help="validation quantity used to pick the best epoch")
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--learning_rate", type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=5e-4)
    g.add_argument("--class_weight", type=str2bool, default=False, help="re-weight the loss if classes are unbalanced")
    g.add_argument("--num_workers", type=int, default=0)

    g = p.add_argument_group("hardware / output")
    g.add_argument("--use_gpu", type=str2bool, default=True, help="uses CUDA if available, otherwise CPU")
    g.add_argument("--gpu", type=int, default=0)
    g.add_argument("--checkpoints", default="checkpoints")
    g.add_argument("--results_dir", default="results")
    return p


def make_setting(args, seed):
    split_tag = os.path.basename(args.split_file or split_file_path(args)).split("__")[1]
    feats = args.node_features.replace(",", "+") or "const"
    if args.extra_node_cols:
        feats += "+" + args.extra_node_cols.replace(",", "+")
    return (f"{args.model_id}_{args.model}_{os.path.splitext(os.path.basename(args.data_path))[0]}"
            f"_{args.graph_type}-w{args.window}_nf-{feats}_ew-{args.edge_weight}{'-undir' if args.undirected else ''}"
            f"_sp-{split_tag}_L{args.num_layers}_H{args.hidden_dim}_{args.readout}_lr{args.learning_rate}_seed{seed}")


def main():
    args = get_parser().parse_args()
    print("Args:", vars(args))

    base_seed = args.seed
    all_metrics = []
    for ii in range(args.itr):
        args.seed = base_seed + ii
        set_seed(args.seed)
        setting = make_setting(args, args.seed)
        exp = Exp_Main(args)
        if args.is_training:
            print(f">>>>>>> start training : {setting} >>>>>>>")
            exp.train(setting)
            print(f">>>>>>> testing : {setting} <<<<<<<")
            all_metrics.append(exp.test(setting))
        else:
            print(f">>>>>>> testing : {setting} <<<<<<<")
            all_metrics.append(exp.test(setting, test=1))

    if len(all_metrics) > 1:
        keys = ["accuracy", "precision", "recall", "f1", "auc"]
        summary = {k: {"mean": float(np.mean([m[k] for m in all_metrics])),
                       "std": float(np.std([m[k] for m in all_metrics], ddof=1))} for k in keys}
        print(f"\n===== {args.model} over {len(all_metrics)} seeds =====")
        for k in keys:
            print(f"  {k:9s} {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
        folder = os.path.join(args.results_dir, make_setting(args, f"{base_seed}x{args.itr}"))
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
    
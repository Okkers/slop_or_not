import os

import numpy as np
import pandas as pd
import torch

from exp.exp_main import Exp_Main
from exp.tools import set_seed
from exp.exp_explainer import GraphExplainer
from main import get_parser, make_setting, str2bool
from vis.utils_explainer import explanation_to_matrices, evaluate_edge_faithfulness_per_motif, analyze_important_nodes, plot_edge_faithfulness, predict_with_edges
from vis.utils_motifs import analyze_motifs

def get_explain_parser():
    # add the explainer arguments to the main parser
    p = get_parser()
    g = p.add_argument_group("explainer")
    g.add_argument("--explain_epochs", type=int, default=200)
    g.add_argument("--node_mask_type", type=str, default="none", choices=["none", "object"])
    g.add_argument("--edge_mask_type", type=str, default="object", choices=["none", "object"])
    g.add_argument("--top_fraction_edges", type=float, default=0.4)
    g.add_argument("--top_k", type=int, default=None)
    g.add_argument("--top_fraction_nodes", type=float, default=0)
    g.add_argument("--top_k_nodes", type=int, default=None)
    return p

def explain_setting(args, setting):
    # rebuild the trained model and load the checkpoint (weights)
    exp = Exp_Main(args)
    ckpt = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    print(f"Checkpoints path {ckpt}")

    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    # create the explainer (edge masks only, node masks are never used)
    explainer = GraphExplainer(
        model=exp.model,
        node_mask_type=None,
        edge_mask_type=None if args.edge_mask_type == "none" else args.edge_mask_type,
        epochs=args.explain_epochs
    )

    # get the predictions that we already have
    folder = os.path.join(args.results_dir, setting)
    print(f"Predictions path {folder}")
    preds = pd.read_csv(os.path.join(folder, "predictions.csv")).set_index("row")

    # get the test data
    test_data, _ = exp._get_data("test")
    n = len(test_data)

    # check if the test data for sure has the same indices as predictions (so test data for model 'exp')
    ids = {int(d.idx) for d in test_data}
    assert ids == set(preds.index)

    # for each test excerpt, get the explanation (averaged over several explainer seeds)
    # and save it to a list of dicts
    results = []
    for i in range(n):
        data = test_data[i].to(exp.device)
        row = int(data.idx)

        # run the explainer several times, each with its own reproducible seed
        runs = []
        for r in range(10):
            print(f"Iteration {r}")
            set_seed(args.seed * 1_000_003 + row * 100 + r)
            explanation = explainer.explain(data)
            em = getattr(explanation, "edge_mask", None)
            if em is not None:
                runs.append(em.detach().cpu())

        if runs:
            stack = torch.stack(runs)  # (runs, E)
            edge_mask = stack.mean(0)
            edge_mask_std = stack.std(0) if len(runs) > 1 else torch.zeros_like(edge_mask)
        else:
            stack = edge_mask = edge_mask_std = None

        results.append({
            "row": row,
            "label": int(data.y),
            "pred": int(preds.loc[row, "pred"]),
            "prob_machine": float(preds.loc[row, "prob_machine"]),
            "x": data.x.cpu(),
            "edge_index": data.edge_index.cpu(),
            "edge_weight": data.edge_weight.cpu(),
            "edge_mask": edge_mask,           # mean over runs, used by all downstream code
            "edge_mask_std": edge_mask_std,   # run-to-run spread per edge
            "edge_mask_runs": stack,          # (runs, E), for stability analysis
            "node_mask": None                 # kept so code reading this key still works
        })

        # explanation.visualize_feature_importance(top_k=10)

    # put the global RNG back into the known state used by the rest of the pipeline
    set_seed(args.seed)

    # save raw explanations
    path = os.path.join(folder, f"explanations_{args.node_mask_type}_{args.edge_mask_type}.pth")
    torch.save(results, path)
    print(f"Saved explanations to {path}")

    # save explanations as matrices for easier analysis
    path_npz = os.path.splitext(path)[0] + ".npz"
    np.savez_compressed(path_npz, **explanation_to_matrices(results, test_data[0].num_nodes, test_data[0].num_features))
    print(f"Saved processed explanations to {path_npz}")

    return results

def load_or_explain(args, setting):
    folder = os.path.join(args.results_dir, setting)

    base = f"explanations_{args.node_mask_type}_{args.edge_mask_type}"

    node_base = ""
    if args.node_mask_type == "attributes":
        node_base = "at"
    if args.node_mask_type == "none":
        node_base = "no"
    if args.node_mask_type == "object":
        node_base = "ob"

    edge_base = ""
    if args.edge_mask_type == "attributes":
            edge_base = "at"
    if args.edge_mask_type == "none":
            edge_base = "no"
    if args.edge_mask_type == "object":
            edge_base = "ob"

    path_base = f"{node_base}{edge_base}"
    path = os.path.join(folder, f"{base}.pth")

    if os.path.exists(path):
        print(f"Loading existing explanations from {path}")

        results = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
    else:
        print("Generating new explanations...")
        results = explain_setting(args, setting)

    return results, path_base

def main():
    args = get_explain_parser().parse_args()
    base_seed = args.seed

    for ii in range(args.itr):
        args.seed = base_seed + ii
        set_seed(args.seed)
        setting = make_setting(args, args.seed)
        folder = os.path.join(args.results_dir, setting)

        results, path_base = load_or_explain(args, setting)

        exp = Exp_Main(args)
        ckpt = os.path.join(args.checkpoints, setting, "checkpoint.pth")
        exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
        exp.model.eval()

        #motif analysis
        if args.edge_mask_type != "none":
            motif_stats = analyze_motifs(results, folder, path_base, fraction=args.top_fraction_edges, top_k=args.top_k, min_present=10)

            df_faithfulness = evaluate_edge_faithfulness_per_motif(
                                    results=results,
                                    folder=folder,
                                    path_base=path_base,
                                    model=exp.model,
                                    device=exp.device,
                                    seed=args.seed,
                                    fraction=args.top_fraction_edges,
                                    top_k=args.top_k,
                                    random_repeats=50)



        #     plot_edge_faithfulness(df_faithfulness, folder)

        # this will not really be used as we focus on edge analysis
        if args.node_mask_type != "none":
            fwords_df = pd.read_csv("data/function_words.txt")
            fwords = fwords_df["word"].tolist()
            analyze_important_nodes(results, folder, path_base, fwords, fraction=args.top_fraction_nodes, top_k=args.top_k_nodes)


        # fig, ax, pos = plot_comparison(results[0], threshold=0.8)
        #
        # plot_comparison(results[1], threshold=0.8, pos=pos)

if __name__ == "__main__":
    main()
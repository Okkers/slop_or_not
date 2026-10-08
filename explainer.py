import os

import numpy as np
import pandas as pd
import torch

from exp.exp_main import Exp_Main
from exp.tools import set_seed
from exp.exp_explainer import GraphExplainer
from main import get_parser, make_setting
from vis.utils_explainer import explanation_to_matrices

def get_explain_parser():
    # add the explainer arguments to the main parser
    p = get_parser()
    g = p.add_argument_group("explainer")
    g.add_argument("--explain_epochs", type=int, default=200)
    g.add_argument("--node_mask_type", type=str, default="attributes", choices=["none", "object", "attributes", "common_attributes"])
    g.add_argument("--edge_mask_type", type=str, default="object", choices=["none", "object", "attributes", "common_attributes"])
    return p

def explain_setting(args, setting):
    # rebuild the trained model and load the checkpoint (weights)
    exp = Exp_Main(args)
    ckpt = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))

    # create the explainer
    explainer = GraphExplainer(
        model=exp.model,
        node_mask_type=None if args.node_mask_type == "none" else args.node_mask_type,
        edge_mask_type=None if args.edge_mask_type == "none" else args.edge_mask_type,
        epochs=args.explain_epochs
    )

    # get the predictions that we already have
    folder = os.path.join(args.results_dir, setting)
    preds = pd.read_csv(os.path.join(folder, "predictions.csv")).set_index("row")

    # get the test data
    test_data, _ = exp._get_data("test")
    n = len(test_data)

    # for each test excerpt, get the explanation and save it to a list of dicts
    results = []
    for i in range(n):
        data = test_data[i].to(exp.device)
        explanation = explainer.explain(data)
        edge_mask = getattr(explanation, "edge_mask", None)
        node_mask = getattr(explanation, "node_mask", None)
        row = int(data.idx)
        results.append({
            "row": row,
            "label": int(data.y),
            "pred": int(preds.loc[row, "pred"]),
            "prob_machine": float(preds.loc[row, "prob_machine"]),
            "x": data.x.cpu(),
            "edge_index": data.edge_index.cpu(),
            "edge_weight": data.edge_weight.cpu(),
            "edge_mask": None if edge_mask is None else edge_mask.cpu(),
            "node_mask": None if node_mask is None else node_mask.cpu()
        })

        # explanation.visualize_feature_importance(top_k=10)

    # save raw explanations
    path = os.path.join(folder, f"explanations_{args.node_mask_type}_{args.edge_mask_type}.pth")
    torch.save(results, path)
    print(f"Saved explanations to {path}")

    # save explanations as matrices for easier analysis
    path_npz = os.path.splitext(path)[0] + ".npz"
    np.savez_compressed(path_npz, **explanation_to_matrices(results, test_data[0].num_nodes, test_data[0].num_features))
    print(f"Saved processed explanations to {path_npz}")

def main():
    args = get_explain_parser().parse_args()
    base_seed = args.seed

    for ii in range(args.itr):
        args.seed = base_seed + ii
        set_seed(args.seed)
        setting = make_setting(args, args.seed)
        explain_setting(args, setting)

if __name__ == "__main__":
    main()
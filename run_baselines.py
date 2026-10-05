"""
Run the SVM baseline on the SAME data, graphs and split as main.py, and add its test
scores to results/results.csv next to the GNNs.

    uv run run_baselines.py --data_path data/persuade_w_wan.csv
    uv run run_baselines.py --data_path data/persuade_w_wan.csv --svm_features freq+graphstats

The SVM kernel and C are picked on the validation split by accuracy; the test split is only
touched once, at the end. All data/split flags of main.py work here too (--split, --window, ...).
"""
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from data_factory.splits import get_splits
from main import get_parser, make_setting
from models.baseline import feature_matrix, make_svm
from vis import utils as wan_utils   # Christina's WAN code
from vis.metrics import append_result, classification_metrics

wan_utils.VERBOSE = False


def load_wans(df, args):
    """Per essay: the WAN, the word counts and the text length (Christina's code in vis/utils.py)."""
    W_list, counts_list, ntok_list = [], [], []
    for row in df.to_dict("records"):
        W, counts, ntok = wan_utils.construct_wan_from_text(
            str(row["output_text"]), windowWidth=args.window, functionWordsFile=args.function_words,
            return_counts=True)
        if args.wan_source == "csv":
            W = json.loads(row["WAN"])
        W_list.append(np.asarray(W, dtype=float))
        counts_list.append(np.asarray(counts, dtype=float))
        ntok_list.append(ntok)
    return W_list, counts_list, ntok_list


def evaluate(name, cfg, scores_val, scores_test, y, sp, args, df, threshold=0.0):
    val_m = classification_metrics(y[sp["val"]], scores_val, threshold=threshold)
    m = classification_metrics(y[sp["test"]], scores_test, threshold=threshold)
    print(f"[{name}] best {cfg} | val acc {val_m['accuracy']:.4f} | TEST acc {m['accuracy']:.4f} "
          f"precision {m['precision']:.4f} recall {m['recall']:.4f} F1 {m['f1']:.4f} AUC {m['auc']:.4f}")
    setting = make_setting(args, args.seed).replace(f"_{args.model}_", f"_{name}_")
    folder = os.path.join(args.results_dir, setting)
    os.makedirs(folder, exist_ok=True)
    pred = pd.DataFrame({"row": sp["test"], "label": y[sp["test"]], "score": scores_test,
                         "pred": (np.asarray(scores_test) >= threshold).astype(int)})
    for col in ("task_prompt", "generator"):
        if col in df.columns:
            pred[col] = df[col].values[sp["test"]]
    pred.to_csv(os.path.join(folder, "predictions.csv"), index=False)
    record = {"setting": setting, "model": name, "model_id": args.model_id,
              "data": os.path.basename(args.data_path), "split": args.split, "graph_type": args.graph_type,
              "window": args.window, "baseline_config": json.dumps(cfg),
              "val_accuracy": val_m["accuracy"], "val_f1": val_m["f1"], "val_auc": val_m["auc"],
              **{f"test_{k}": v for k, v in m.items()}}
    with open(os.path.join(folder, "metrics.json"), "w") as f:
        json.dump(record, f, indent=2)
    append_result(record, args.results_dir)
    return m


def pick_best(candidates, fit_score, y_val):
    """candidates: list of configs; fit_score(cfg) -> (val_scores, test_scores). Select by val accuracy."""
    best = None
    for cfg in candidates:
        sv, st = fit_score(cfg)
        acc = classification_metrics(y_val, sv, threshold=0.0)["accuracy"]
        if best is None or acc > best[0]:
            best = (acc, cfg, sv, st)
    return best[1], best[2], best[3]


def main():
    p = get_parser()
    g = p.add_argument_group("SVM baseline")
    g.add_argument("--svm_features", default="freq,freq+graphstats,freq+wan",
                   help="comma-separated feature sets, each a '+'-combination of freq, wan, graphstats; "
                        "one SVM (and one results row) per set")
    g.add_argument("--svm_C", default="0.01,0.1,1,10")
    g.add_argument("--svm_kernels", default="linear,rbf")
    args = p.parse_args()
    args.model = "BASELINE"

    df = pd.read_csv(args.data_path)
    y = df["label"].astype(int).values
    sp = get_splits(df, args)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")
    t0 = time.time()
    W, counts, ntok = load_wans(df, args)
    print(f"built {len(W)} WANs in {time.time() - t0:.1f}s")
    Cs = [float(c) for c in args.svm_C.split(",")]

    for feat_set in args.svm_features.split(","):
        X = feature_matrix(W, counts, ntok, feat_set)

        def fit_svm(cfg):
            clf = make_svm(cfg["kernel"], cfg["C"]).fit(X[tr], y[tr])
            return clf.decision_function(X[va]), clf.decision_function(X[te])
        cands = [{"features": feat_set, "kernel": k, "C": c}
                 for k, c in itertools.product(args.svm_kernels.split(","), Cs)]
        cfg, sv, st = pick_best(cands, fit_svm, y[va])
        evaluate(f"SVM-{feat_set}", cfg, sv, st, y, sp, args, df)


if __name__ == "__main__":
    main()
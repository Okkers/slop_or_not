"""
Run the non-GNN baselines on the SAME data, graphs and split as main.py, and add their
test scores to results/results.csv next to the GNNs.

    uv run run_baselines.py --data_path data/persuade_w_wan.csv                     # all baselines
    uv run run_baselines.py --data_path data/persuade_w_wan.csv --baseline SVM --svm_features freq+graphstats

Hyper-parameters (SVM kernel and C, WL iterations / edge threshold / C) are picked on the
validation split by accuracy; the test split is only touched once, at the end.
All data/split flags of main.py work here too (--split, --window, --test_groups, ...).
"""
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from data_factory.splits import get_splits
from data_factory.wan import build_wan, load_function_words
from main import get_parser, make_setting
from models.baseline import (RelEntClassifier, feature_matrix, make_svm, make_wl_svm, wl_features)
from vis.metrics import append_result, classification_metrics


def load_wans(df, args):
    fwords = load_function_words(args.function_words)
    W_list, raw_list, counts_list, ntok_list = [], [], [], []
    for row in df.to_dict("records"):
        raw, counts, ntok = build_wan(str(row["output_text"]), fwords, window=args.window, normalize=False)
        rs = raw.sum(1, keepdims=True)
        W = np.divide(raw, rs, out=np.zeros_like(raw), where=rs != 0)
        if args.wan_source == "csv":
            W = np.asarray(json.loads(row["WAN"]), dtype=float)
        W_list.append(W); raw_list.append(raw); counts_list.append(counts); ntok_list.append(ntok)
    return W_list, raw_list, counts_list, ntok_list


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
    g = p.add_argument_group("baselines")
    g.add_argument("--baseline", default="all", help="SVM | WL | RelEnt | all")
    g.add_argument("--svm_features", default="freq,freq+graphstats,freq+wan",
                   help="comma-separated feature sets, each a '+'-combination of freq, wan, graphstats; "
                        "one SVM (and one results row) per set")
    g.add_argument("--svm_C", default="0.01,0.1,1,10")
    g.add_argument("--svm_kernels", default="linear,rbf")
    g.add_argument("--wl_iterations", default="1,2")
    g.add_argument("--wl_thresholds", default="0.1,0.25,0.4",
                   help="keep WAN edges with weight >= threshold (sparser graphs -> WL labels repeat more)")
    args = p.parse_args()
    args.model = "BASELINE"

    df = pd.read_csv(args.data_path)
    y = df["label"].astype(int).values
    sp = get_splits(df, args)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")
    t0 = time.time()
    W, raw, counts, ntok = load_wans(df, args)
    print(f"built {len(W)} WANs in {time.time() - t0:.1f}s")
    which = ["SVM", "WL", "RelEnt"] if args.baseline == "all" else [args.baseline]
    Cs = [float(c) for c in args.svm_C.split(",")]

    if "SVM" in which:
        for feat_set in args.svm_features.split(","):
            X = feature_matrix(W, counts, ntok, feat_set)

            def fit_svm(cfg):
                clf = make_svm(cfg["kernel"], cfg["C"]).fit(X[tr], y[tr])
                return clf.decision_function(X[va]), clf.decision_function(X[te])
            cands = [{"features": feat_set, "kernel": k, "C": c}
                     for k, c in itertools.product(args.svm_kernels.split(","), Cs)]
            cfg, sv, st = pick_best(cands, fit_svm, y[va])
            evaluate(f"SVM-{feat_set}", cfg, sv, st, y, sp, args, df)

    if "WL" in which:
        feats = {}

        def fit_wl(cfg):
            key = (cfg["h"], cfg["threshold"])
            if key not in feats:
                feats[key] = wl_features(W, counts, iterations=cfg["h"], threshold=cfg["threshold"])
            Xw = feats[key]
            clf = make_wl_svm(cfg["C"]).fit(Xw[tr], y[tr])
            return clf.decision_function(Xw[va]), clf.decision_function(Xw[te])
        cands = [{"h": int(h), "threshold": float(t), "C": c}
                 for h, t, c in itertools.product(args.wl_iterations.split(","), args.wl_thresholds.split(","), Cs)]
        cfg, sv, st = pick_best(cands, fit_wl, y[va])
        evaluate("WL-kernel-SVM", cfg, sv, st, y, sp, args, df)

    if "RelEnt" in which:
        clf = RelEntClassifier().fit([raw[i] for i in tr], y[tr])
        sv = clf.decision_function([W[i] for i in va])
        st = clf.decision_function([W[i] for i in te])
        evaluate("WAN-RelEnt", {"rule": "nearest class profile"}, sv, st, y, sp, args, df)


if __name__ == "__main__":
    main()

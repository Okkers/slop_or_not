"""Classification metrics. F1 is the primary metric (proposal, Section 2); label 1 = machine = positive class."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y_true, prob_machine, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob_machine = np.asarray(prob_machine, dtype=float)
    y_pred = (prob_machine >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob_machine) if len(np.unique(y_true)) == 2 else float("nan"),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def append_result(record: dict, results_dir: str):
    """Add one row to <results_dir>/results.csv (GNNs and baselines share this table;
    columns are merged, so rows with different fields line up)."""
    import os

    import pandas as pd
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "results.csv")
    new = pd.DataFrame([record])
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)

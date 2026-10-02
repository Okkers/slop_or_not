import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data_factory.data_provider import data_provider
from exp.exp_basic import Exp_Basic
from exp.tools import EarlyStopping
from models import MODELS
from vis.metrics import append_result, classification_metrics


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        self._data = {}
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # the input size depends on the chosen node features, so read it off the data
        self.args.in_dim = self._get_data("train")[0].num_node_features
        self.args.num_classes = 2
        if self.args.model not in MODELS:
            raise ValueError(f"unknown model '{self.args.model}', choose from {list(MODELS)}")
        model = MODELS[self.args.model].Model(self.args).float()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{self.args.model}: {n_params:,} trainable parameters, input dim {self.args.in_dim}")
        return model

    def _get_data(self, flag):
        if flag not in self._data:
            self._data[flag] = data_provider(self.args, flag)
        return self._data[flag]

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def _select_criterion(self, train_data=None):
        weight = None
        if self.args.class_weight and train_data is not None:
            counts = np.bincount(train_data.labels, minlength=2).astype(float)
            weight = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=self.device)
            print(f"class weights: {weight.tolist()}")
        return nn.CrossEntropyLoss(weight=weight)

    def _predict(self, batch):
        return self.model(batch.x, batch.edge_index, edge_weight=batch.edge_weight, batch=batch.batch)

    @torch.no_grad()
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss, idx, ys, probs = 0.0, [], [], []
        for batch in vali_loader:
            batch = batch.to(self.device)
            logits = self._predict(batch)
            total_loss += criterion(logits, batch.y).item() * batch.num_graphs
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
            ys.append(batch.y.cpu())
            idx.append(batch.idx.cpu())
        probs, ys, idx = torch.cat(probs).numpy(), torch.cat(ys).numpy(), torch.cat(idx).numpy()
        self.model.train()
        return total_loss / len(vali_data), classification_metrics(ys, probs), (idx, ys, probs)

    def train(self, setting):
        train_data, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        early_stopping = EarlyStopping(patience=self.args.patience, metric=self.args.es_metric,
                                       verbose=self.args.verbose)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(train_data)

        history = []
        for epoch in range(self.args.train_epochs):
            t0 = time.time()
            self.model.train()
            total = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                model_optim.zero_grad()
                loss = criterion(self._predict(batch), batch.y)
                loss.backward()
                model_optim.step()
                total += loss.item() * batch.num_graphs
            train_loss = total / len(train_data)
            vali_loss, vm, _ = self.vali(vali_data, vali_loader, criterion)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": vali_loss,
                            "val_f1": vm["f1"], "val_acc": vm["accuracy"], "val_auc": vm["auc"]})
            if self.args.verbose and (epoch == 0 or (epoch + 1) % self.args.log_every == 0):
                print(f"Epoch {epoch + 1:3d} | {time.time() - t0:5.2f}s | train loss {train_loss:.4f} | "
                      f"val loss {vali_loss:.4f} | val acc {vm['accuracy']:.4f} | val F1 {vm['f1']:.4f}")
            early_stopping(vali_loss, vm, self.model, path, epoch + 1)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1} (best epoch {early_stopping.best_epoch})")
                break

        pd.DataFrame(history).to_csv(os.path.join(path, "history.csv"), index=False)
        self.best_epoch = early_stopping.best_epoch
        self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth"), map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data("test")
        vali_data, vali_loader = self._get_data("val")
        if test:
            print("loading model")
            ckpt = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        criterion = self._select_criterion()

        _, val_m, _ = self.vali(vali_data, vali_loader, criterion)
        test_loss, m, (idx, ys, probs) = self.vali(test_data, test_loader, criterion)
        print(f"TEST  acc {m['accuracy']:.4f} | precision {m['precision']:.4f} | recall {m['recall']:.4f} | "
              f"F1 {m['f1']:.4f} | AUC {m['auc']:.4f}   (TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']})")

        folder = os.path.join(self.args.results_dir, setting)
        os.makedirs(folder, exist_ok=True)

        # per-excerpt predictions, mapped back to the CSV rows (for error analysis / explanations)
        pred = pd.DataFrame({"row": idx, "label": ys, "prob_machine": probs, "pred": (probs >= 0.5).astype(int)})
        for col in ("task_prompt", "generator"):
            if col in test_data.df.columns:
                pred[col] = test_data.df[col].values[idx]
        pred.to_csv(os.path.join(folder, "predictions.csv"), index=False)

        record = {"setting": setting, "model": self.args.model, "model_id": self.args.model_id,
                  "data": os.path.basename(self.args.data_path), "split": self.args.split,
                  "graph_type": self.args.graph_type, "window": self.args.window,
                  "node_features": self.args.node_features, "edge_weight": self.args.edge_weight,
                  "undirected": int(self.args.undirected), "num_layers": self.args.num_layers,
                  "hidden_dim": self.args.hidden_dim, "readout": self.args.readout,
                  "root_weight": int(self.args.root_weight),
                  "learning_rate": self.args.learning_rate, "seed": self.args.seed,
                  "best_epoch": getattr(self, "best_epoch", None),
                  **{f"val_{k}": v for k, v in val_m.items() if k in ("accuracy", "f1", "auc")},
                  **{f"test_{k}": v for k, v in m.items()}}
        with open(os.path.join(folder, "metrics.json"), "w") as f:
            json.dump({"record": record, "args": {k: v for k, v in vars(self.args).items()
                                                  if isinstance(v, (int, float, str, bool, type(None)))}}, f, indent=2)
        if not test:   # a re-test of an existing checkpoint would duplicate the row
            append_result(record, self.args.results_dir)
        return m

    @torch.no_grad()
    def predict(self, setting, load=True):
        """Score every row of --data_path with a trained checkpoint -> results/<setting>/all_predictions.csv"""
        pred_data, pred_loader = self._get_data("pred")
        if load:
            ckpt = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        _, m, (idx, ys, probs) = self.vali(pred_data, pred_loader, self._select_criterion())
        folder = os.path.join(self.args.results_dir, setting)
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame({"row": idx, "label": ys, "prob_machine": probs}).to_csv(
            os.path.join(folder, "all_predictions.csv"), index=False)
        return m

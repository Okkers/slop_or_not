"""
Load a trained model back from disk (for explanations, error analysis, notebooks).

    from exp.load import load_trained
    model, args, test_set = load_trained("results/<setting>")
    model(g.x, g.edge_index, edge_weight=g.edge_weight)        # logits for one graph

Works for every model in models/ because main.py stores the run's arguments in
results/<setting>/metrics.json and the weights in checkpoints/<setting>/checkpoint.pth.
"""
import json
import os
from argparse import Namespace

import torch

from data_factory.data_loader import Dataset_MTT_Detection
from models import MODELS


def load_trained(result_folder: str, checkpoints: str | None = None, flag: str = "test", device: str = "cpu"):
    with open(os.path.join(result_folder, "metrics.json")) as f:
        args = Namespace(**json.load(f)["args"])
    setting = os.path.basename(os.path.normpath(result_folder))
    ckpt = os.path.join(checkpoints or args.checkpoints, setting, "checkpoint.pth")
    model = MODELS[args.model].Model(args)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    data = Dataset_MTT_Detection(args, flag)
    return model, args, data

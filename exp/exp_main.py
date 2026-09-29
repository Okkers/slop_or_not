from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'GAT': NotImplementedError,
            "GNN": NotImplementedError
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, x, y):
        ### Helper function
        return 

    def vali(self, vali_data, vali_loader, criterion):
        ### TO-DO => Implement Validation setting

        return 

    def train(self, setting):
        ### TO-DO => Implement Training setting

        return

    def test(self, setting, test=0):
        ### TO-DO => Implement Testing setting

        return

    def predict(self, setting, load=False):
        ### TO-DO => Implement Inference

        return
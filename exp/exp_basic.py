import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self, flag):
        pass

    def vali(self, *args, **kwargs):
        pass

    def train(self, setting):
        pass

    def test(self, setting, test=0):
        pass

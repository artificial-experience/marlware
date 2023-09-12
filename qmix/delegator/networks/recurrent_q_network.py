import numpy as np
import torch as T
import torch.functional as F
import torch.nn as nn
import torch.optim as optim


class DRQN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

    def construct_network(self):
        pass

    def forward(self, observation):
        pass

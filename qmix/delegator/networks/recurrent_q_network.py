import numpy as np
import torch as T
import torch.functional as F
import torch.nn as nn
import torch.optim as optim


class DQRN(nn.Module):
    def __init__(self, config):
        super(DRQN, self).__init__()
        self._config = config

    def construct_network(self):
        pass

    def forward(self, observation):
        pass

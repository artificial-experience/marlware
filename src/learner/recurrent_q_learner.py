from functools import partialmethod

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F


class RecurrentQLearner:
    """
    Deep recurrent q-network agent
    Single agent instance that has a shared drqn network

    Args:
        param: [conf]: agent hyperparameter configuration
        param: [unique_id]: agents unique identifier
    """

    def __init__(self, conf: OmegaConf, identifier: F.one_hot):
        self._conf = conf

        # agent unique id - one hot
        self._identifier = identifier

        # internal attrs
        self._eval_net = None
        self._target_net = None

    def share_networks(
        self, eval_net: torch.nn.Module, target_net: torch.nn.Module
    ) -> None:
        """assign shared networks to learner"""
        self._eval_net = eval_net
        self._target_net = target_net

    def estimate_q_value(self, feed: torch.Tensor, use_target=False) -> torch.Tensor:
        """estimate q value given feed tensor"""
        q_vals = self._target_net(feed) if use_target else self._eval_net(feed)
        return q_vals

    # ---- ---- ---- ---- ---- #
    # --- Partial Methods ---- #
    # ---- ---- ---- ---- ---- #

    estimate_eval_q = partialmethod(estimate_q_value, use_target=False)
    estimate_target_q = partialmethod(estimate_q_value, use_target=True)

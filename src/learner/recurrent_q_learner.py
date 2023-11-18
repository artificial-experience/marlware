from functools import partialmethod
from typing import Dict

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

    @property
    def one_hot_identifier(self) -> F.one_hot:
        return self._identifier

    def share_networks(
        self, eval_net: torch.nn.Module, target_net: torch.nn.Module
    ) -> None:
        """assign shared networks to learner"""
        self._eval_net = eval_net
        self._target_net = target_net

    def estimate_q_value(
        self, feed: Dict[str, torch.Tensor], use_target=False
    ) -> torch.Tensor:
        """estimate q value given feed tensor"""
        observations = feed.get("observations", None)
        avail_actions = feed.get("avail_actions", None)

        bs, n_agents, n_q_values = avail_actions.shape

        agent_identifier_one_hot = self.one_hot_identifier.view(1, -1)
        agent_identifier_one_hot = agent_identifier_one_hot.repeat(bs, 1)
        serialized_identifier = torch.argmax(agent_identifier_one_hot)

        # get current agent's observations slices
        agent_observations = observations[:, serialized_identifier, :]

        prepared_feed = torch.cat(
            [agent_observations, agent_identifier_one_hot], dim=-1
        )
        prepared_feed = prepared_feed.view(bs, -1)

        # zero hidden state before every update
        initial_hidden = (
            self._target_net.init_hidden_state(bs)
            if use_target
            else self._eval_net.init_hidden_state(bs)
        )

        q_vals, hidden = (
            self._target_net(prepared_feed, initial_hidden)
            if use_target
            else self._eval_net(prepared_feed, initial_hidden)
        )
        return q_vals, hidden

    # ---- ---- ---- ---- ---- #
    # --- Partial Methods ---- #
    # ---- ---- ---- ---- ---- #

    estimate_eval_q = partialmethod(estimate_q_value, use_target=False)
    estimate_target_q = partialmethod(estimate_q_value, use_target=True)

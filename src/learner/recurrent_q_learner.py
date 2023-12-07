from functools import partialmethod
from typing import Dict

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from src.transforms import OneHotTransform
from src.util import methods
from src.util.constants import AttrKey


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
        data_attr = AttrKey.data

        observations = feed[data_attr._OBS.value]
        avail_actions = feed[data_attr._AVAIL_ACTIONS.value]
        actions = feed[data_attr._ACTIONS.value]

        bs, time, n_agents, n_q_values = avail_actions.shape

        # ensure proper shape of tensors - without time as this dimension is 1
        observations = observations.view(bs, n_agents, -1)
        actions = actions.view(bs, n_agents, -1)

        # prepare agent actions
        one_hot = OneHotTransform(n_q_values)
        one_hot_actions = one_hot.transform(actions).view(bs, n_agents, -1)

        agent_identifier_one_hot = self.one_hot_identifier.view(1, -1)
        agent_identifier_one_hot = agent_identifier_one_hot.repeat(bs, 1)
        serialized_identifier = torch.argmax(agent_identifier_one_hot)

        # get current agent's observations slices
        agent_observations = observations[:, serialized_identifier, :]
        agent_one_hot_actions = one_hot_actions[:, serialized_identifier, :]

        prepared_feed = torch.cat(
            [agent_observations, agent_identifier_one_hot, agent_one_hot_actions],
            dim=-1,
        )
        prepared_feed = prepared_feed.view(bs, -1)

        q_vals = (
            self._target_net(prepared_feed)
            if use_target
            else self._eval_net(prepared_feed)
        )

        return q_vals

    # ---- ---- ---- ---- ---- #
    # @ -> Partial Methods
    # ---- ---- ---- ---- ---- #

    estimate_eval_q = partialmethod(estimate_q_value, use_target=False)
    estimate_target_q = partialmethod(estimate_q_value, use_target=True)

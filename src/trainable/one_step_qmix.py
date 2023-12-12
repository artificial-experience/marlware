from functools import partialmethod
from typing import Dict
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from .proto import ProtoQmix
from src.registry import register_trainable
from src.util.constants import AttrKey


@register_trainable
class OneStepQmix(ProtoQmix):
    """
    One-Step TD version for  QMIX: Monotonic Value Function Factorisation
    for Deep Multi-Agent Reinforcement Learning

    Args:
        :param [hypernet_conf]: hypernetwork configuration
        :param [mixer_conf]: mixer head configuration

    Derived State:
        :param [eval_mixer]: evaluation mixing network instance
        :param [target_mixer]: frozen instance of network used for target calculation

    """

    def __init__(self, hypernet_conf: OmegaConf, mixer_conf: OmegaConf) -> None:
        super().__init__(hypernet_conf, mixer_conf)

    def ensemble_trainable(
        self,
        n_agents: int,
        observation_dim: int,
        state_dim: int,
        gamma: float,
        *,
        seed: Optional[int] = None
    ) -> None:
        super().ensemble_trainable(
            n_agents, observation_dim, state_dim, gamma, seed=seed
        )

    def factorize_q_vals(
        self, agent_qs: torch.Tensor, states: torch.Tensor, use_target: bool = False
    ) -> torch.Tensor:
        """takes batch and computes factorised q-value"""
        factorized_qs = (
            self._target_mixer(agent_qs, states)
            if use_target
            else self._eval_mixer(agent_qs, states)
        )
        return factorized_qs

    def synchronize_target_net(self, tau: float = 1.0):
        """copy weights from eval net to target net using tau temperature.

        for tau = 1.0, this performs a hard update.
        for 0 < tau < 1.0, this performs a soft update.
        """
        for target_param, eval_param in zip(
            self._target_mixer.parameters(), self._eval_mixer.parameters()
        ):
            target_param.data.copy_(
                tau * eval_param.data + (1 - tau) * target_param.data
            )

    def calculate_loss(
        self,
        feed: Dict[str, torch.Tensor],
        eval_q_vals: torch.Tensor,
        target_q_vals: torch.Tensor,
    ) -> torch.Tensor:
        """use partial methods to calculate criterion loss between eval and target"""

        # ---- ---- ---- ---- ---- #
        # @ -> Get Data
        # ---- ---- ---- ---- ---- #

        data_attr = AttrKey.data

        actions = feed[data_attr._ACTIONS.value]
        state = feed[data_attr._STATE.value]
        reward = feed[data_attr._REWARD.value]
        terminated = feed[data_attr._TERMINATED.value]
        avail_actions = feed[data_attr._AVAIL_ACTIONS.value]
        mask = feed[data_attr._FILLED.value]

        # ---- ---- ---- ---- ---- #
        # @ -> Transform Data
        # ---- ---- ---- ---- ---- #

        actions = actions[:, :-1]
        reward = reward[:, :-1]
        terminated = terminated[:, :-1].float()
        mask = mask[:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = avail_actions[:, 1:]

        eval_state = state[:, :-1]
        target_state = state[:, 1:]

        # ---- ---- ---- ---- ---- #
        # @ -> Prepare Q-Vals
        # ---- ---- ---- ---- ---- #

        chosen_action_q_vals = torch.gather(eval_q_vals, dim=3, index=actions).squeeze(
            3
        )

        target_q_vals[avail_actions == 0] = -9999999
        target_max_q_vals = target_q_vals.max(dim=3)[0]

        # ---- ---- ---- ---- ---- #
        # @ -> Factorize Q-Vals
        # ---- ---- ---- ---- ---- #

        eval_factorized_values = self.factorize_eval_q_vals(
            chosen_action_q_vals, eval_state
        )
        target_factorized_values = self.factorize_target_q_vals(
            target_max_q_vals, target_state
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Calculate TD Target
        # ---- ---- ---- ---- ---- #

        target = reward + self._gamma * (1 - terminated) * target_factorized_values
        target = target.detach()

        td_error = eval_factorized_values - target
        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # L2 Loss over actual data
        loss = 0.5 * (masked_td_error**2).sum() / mask.sum()

        return loss

    # ---- ---- ---- ---- ---- #
    # @ -> Partial Methods
    # ---- ---- ---- ---- ---- #

    factorize_eval_q_vals = partialmethod(factorize_q_vals, use_target=False)
    factorize_target_q_vals = partialmethod(factorize_q_vals, use_target=True)

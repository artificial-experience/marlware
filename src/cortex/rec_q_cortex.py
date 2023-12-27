from functools import partialmethod
from typing import Dict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .proto import ProtoCortex
from src.util.constants import AttrKey


class RecQCortex(ProtoCortex):
    """
    Multi-agent coordination
    NOTE: Shared parameters implementation

    Args:
        :param [model_conf]: mac configuration OmegaConf
            contains configuration for learners
        :param [exp_conf]: mac configuration OmegaConf
            contains configuration for heuristics [ policy ]

    Derived State:
        :param [policy]: policy to be followed by each agent in mac
    """

    def __init__(self, model_conf: OmegaConf, exp_conf: OmegaConf) -> None:
        super().__init__(model_conf, exp_conf)

    def ensemble_cortex(
        self,
        n_agents: int,
        n_actions: int,
        obs_shape: tuple,
        *,
        seed: Optional[int] = None
    ) -> None:
        """create heuristic, networks and ensemble N learners"""
        super().ensemble_cortex(n_agents, n_actions, obs_shape, seed=seed)

    def infer_actions(
        self,
        data: dict,
        rollout_timestep: int,
        env_timestep: int,
        evaluate: bool = False,
    ) -> np.ndarray:
        """get feed and compute agents actions to take in the environment"""

        data_attr = AttrKey.data

        # prepare feed and expand tensor to 4d
        observations = data[data_attr._OBS.value][rollout_timestep, :].expand(
            1, 1, -1, -1
        )
        avail_actions = data[data_attr._AVAIL_ACTIONS.value][
            rollout_timestep, :
        ].expand(1, 1, -1, -1)
        actions = data[data_attr._ACTIONS.value][rollout_timestep, :].expand(
            1, 1, -1, -1
        )

        # prepare fed for the agent
        feed = {
            data_attr._OBS.value: observations,
            data_attr._AVAIL_ACTIONS.value: avail_actions,
            data_attr._ACTIONS.value: actions,
        }

        bs, ts, n_agents, n_q_values = avail_actions.shape

        t_multi_agent_q_vals = self.estimate_eval_q_vals(feed)
        t_multi_agent_q_vals = t_multi_agent_q_vals.detach()

        # ts X n_agents X n_q_vals
        avail_actions = avail_actions.view(-1, n_agents, n_q_values)

        decided_actions = (
            self._policy.decide_actions_greedily(
                t_multi_agent_q_vals, avail_actions, env_timestep
            )
            if evaluate
            else self._policy.decide_actions_epsilon_greedily(
                t_multi_agent_q_vals, avail_actions, env_timestep
            )
        )
        return decided_actions.detach().numpy()

    def estimate_q_vals(
        self, feed: Dict[str, torch.Tensor], use_target: bool = False
    ) -> torch.Tensor:
        """either use eval or target net to estimate q values"""
        multi_agent_q_vals = []
        for agent in self._agents:
            q_vals = (
                agent.estimate_target_q(feed)
                if use_target
                else agent.estimate_eval_q(feed)
            )

            multi_agent_q_vals.append(q_vals)

        # n_agents X batch_size X n_q_vals
        t_multi_agent_q_vals = torch.stack(multi_agent_q_vals, dim=1)
        return t_multi_agent_q_vals

    # ---- ---- ---- ---- ---- #
    # @ -> Partial Methods
    # ---- ---- ---- ---- ---- #

    infer_eps_greedy_actions = partialmethod(infer_actions, evaluate=False)
    infer_greedy_actions = partialmethod(infer_actions, evaluate=True)

    estimate_eval_q_vals = partialmethod(estimate_q_vals, use_target=False)
    estimate_target_q_vals = partialmethod(estimate_q_vals, use_target=True)

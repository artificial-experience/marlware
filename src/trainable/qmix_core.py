import copy
import random
from functools import partialmethod
from typing import Dict
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from src.abstract import ProtoTrainable
from src.net import QMixer
from src.registry import register_trainable
from src.util.constants import AttrKey


@register_trainable
class QmixCore(ProtoTrainable):
    """
    Core implementation of QMIX: Monotonic Value Function Factorisation
    for Deep Multi-Agent Reinforcement Learning

    Args:
        :param [hypernet_conf]: hypernetwork configuration
        :param [mixer_conf]: mixer head configuration

    Internal State:
        :param [eval_mixer]: evaluation mixing network instance
        :param [target_mixer]: frozen instance of network used for target calculation

    """

    def __init__(self, hypernet_conf: OmegaConf, mixer_conf: OmegaConf) -> None:
        self._hypernet_conf = hypernet_conf
        self._mixer_conf = mixer_conf

        # internal attrs
        self._eval_mixer = None
        self._target_mixer = None

        # loss calculation
        self._gamma = None

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def ensemble_trainable(
        self,
        n_agents: int,
        observation_dim: int,
        state_dim: int,
        gamma: float,
        *,
        seed: Optional[int] = None
    ) -> None:
        self._rnd_seed(seed=seed)

        # ---- ---- ---- ---- ---- ---- #
        # @ -> Prepare Mixers
        # ---- ---- ---- ---- ---- ---- #

        hypernet_embed_dim = self._hypernet_conf.embedding_dim
        mixer_embed_dim = self._mixer_conf.embedding_dim
        n_hypernet_layers = self._hypernet_conf.n_layers
        self._eval_mixer = QMixer(
            hypernet_embed_dim=hypernet_embed_dim,
            mixer_embed_dim=mixer_embed_dim,
            n_hypernet_layers=n_hypernet_layers,
        )
        self._eval_mixer.integrate_network(n_agents, state_dim, seed=seed)

        # deepcopy eval network structure for frozen mixer networl
        self._target_mixer = copy.deepcopy(self._eval_mixer)

        # ---- ---- ---- ---- ---- ---- #
        # @ -> Prepare Hyperparams
        # ---- ---- ---- ---- ---- ---- #

        self._gamma = gamma

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

    def parameters(self):
        """return hypernet and mixer optimization params"""
        return self._eval_mixer.parameters()

    def move_to_cuda(self):
        """move models to cuda device"""
        self._eval_mixer.cuda()
        self._target_mixer.cuda()

    def calculate_loss(
        self,
        feed: Dict[str, torch.Tensor],
        eval_q_vals: torch.Tensor,
        target_q_vals: torch.Tensor,
    ) -> torch.Tensor:
        """use partial methods to calcualte criterion loss between eval and target"""

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
        td_error = eval_factorized_values - target.detach()
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

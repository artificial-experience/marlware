import random
from functools import partialmethod
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical

from src.heuristic.schedule import DecayThenFlatSchedule


class EpsilonGreedy:
    def __init__(
        self, epsilon_start, epsilon_finish, epsilon_anneal_time, test_noise=0.0
    ):
        self.schedule = DecayThenFlatSchedule(
            epsilon_start, epsilon_finish, epsilon_anneal_time, decay="linear"
        )
        self.epsilon = self.schedule.eval(0)
        self.test_noise = test_noise

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def ensemble_policy(self, *, seed: Optional[int] = None):
        """create policy and assign seed"""
        self._rnd_seed(seed=seed)

    def decide_actions(self, agent_inputs, avail_actions, timestep, test_mode=False):
        # Epsilon adjustment based on timestep or test mode
        self.epsilon = (
            self.schedule.eval(timestep) if not test_mode else self.test_noise
        )

        # Mask unavailable actions in Q-values
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")

        # Generate random numbers for the entire batch
        random_numbers = torch.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()

        # Sample random actions where needed
        # Note: Using Categorical distribution assumes that avail_actions are probabilities
        random_actions = Categorical(avail_actions.float()).sample().long()

        # Select actions: random where pick_random is 1, best action otherwise
        picked_actions = (
            pick_random * random_actions
            + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        )

        return picked_actions

    # ---- ---- ---- ---- ---- #
    # @ -> Partial Methods
    # ---- ---- ---- ---- ---- #

    decide_actions_epsilon_greedily = partialmethod(decide_actions, test_mode=False)
    decide_actions_greedily = partialmethod(decide_actions, test_mode=True)

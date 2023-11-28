import random
from functools import partialmethod
from typing import Optional

import numpy as np
import torch

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
        self.epsilon = (
            self.schedule.eval(timestep) if not test_mode else self.test_noise
        )

        bs, n_agents, n_q_values = agent_inputs.size()

        # Clone and mask unavailable actions in Q-values
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")

        # Initialize tensor to store chosen actions
        chosen_actions = torch.empty(bs, n_agents, dtype=torch.int64)

        # Decide actions for each agent
        for batch_idx in range(bs):
            for agent_idx in range(n_agents):
                if (
                    torch.rand(1).item() < self.epsilon
                ):  # Exploration: Choose a random action
                    available_actions = avail_actions[batch_idx, agent_idx].nonzero(
                        as_tuple=True
                    )[0]
                    if len(available_actions) > 0:
                        chosen_actions[batch_idx, agent_idx] = available_actions[
                            torch.randint(0, len(available_actions), (1,))
                        ]
                    else:
                        chosen_actions[batch_idx, agent_idx] = torch.tensor(
                            -1
                        )  # Handle case with no available actions
                else:  # Exploitation: Choose the best action
                    agent_q_values = masked_q_values[batch_idx, agent_idx]
                    chosen_action = agent_q_values.max(0)[
                        1
                    ].item()  # Get the index of the max Q-value
                    chosen_actions[batch_idx, agent_idx] = chosen_action

        return chosen_actions

    # ---- ---- ---- ---- ---- #
    # @ -> Partial Methods
    # ---- ---- ---- ---- ---- #

    decide_actions_epsilon_greedily = partialmethod(decide_actions, test_mode=False)
    decide_actions_greedily = partialmethod(decide_actions, test_mode=True)

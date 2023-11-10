from functools import partialmethod

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

    def decide_actions(self, agent_inputs, avail_actions, timestep, test_mode=False):
        # Update epsilon according to the schedule or set to test noise level if in test mode
        self.epsilon = (
            self.schedule.eval(timestep) if not test_mode else self.test_noise
        )

        # Mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")

        # Generate a tensor of random numbers for each agent
        random_numbers = torch.rand(agent_inputs.shape[0])
        pick_random = (random_numbers < self.epsilon).long()

        # Sample random actions for each agent
        random_actions = Categorical(avail_actions.float()).sample()

        # Find the actions with the maximum Q-value
        best_actions = masked_q_values.max(dim=1)[1]

        # Combine random and best actions depending on the value of pick_random
        picked_actions = pick_random * random_actions + (1 - pick_random) * best_actions

        return picked_actions

    # ---- ---- ---- ---- ---- #
    # --- Partial Methods ---- #
    # ---- ---- ---- ---- ---- #

    decide_actions_epsilon_greedily = partialmethod(decide_actions, test_mode=False)
    decide_actions_greedily = partialmethod(decide_actions, test_mode=True)

from functools import partialmethod

import numpy as np
import torch
from torch.distributions import Categorical


class DecayThenFlatSchedule:
    def __init__(self, start, finish, time_length, decay="exp"):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay == "exp":
            self.exp_scaling = (
                (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1
            )

    def eval(self, T):
        if self.decay == "linear":
            return max(self.finish, self.start - self.delta * T)
        elif self.decay == "exp":
            return min(self.start, max(self.finish, np.exp(-T / self.exp_scaling)))


class EpsilonGreedyActionSelector:
    def __init__(
        self, epsilon_start, epsilon_finish, epsilon_anneal_time, test_noise=0.0
    ):
        self.schedule = DecayThenFlatSchedule(
            epsilon_start, epsilon_finish, epsilon_anneal_time, decay="linear"
        )
        self.epsilon = self.schedule.eval(0)
        self.test_noise = test_noise

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Update epsilon according to the schedule or set to test noise level if in test mode
        self.epsilon = self.schedule.eval(t_env) if not test_mode else self.test_noise

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

    select_action_train = partialmethod(select_action, test_mode=False)
    select_action_test = partialmethod(select_action, test_mode=True)

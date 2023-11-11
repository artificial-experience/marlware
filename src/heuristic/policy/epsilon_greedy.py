from functools import partialmethod

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

    def decide_actions(self, agent_inputs, avail_actions, timestep, test_mode=False):
        self.epsilon = (
            self.schedule.eval(timestep) if not test_mode else self.test_noise
        )

        n_agents, bs, n_q_values = agent_inputs.size()

        # Clone and mask unavailable actions in Q-values
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")

        # Initialize tensor to store chosen actions
        chosen_actions = torch.empty(n_agents, dtype=torch.long)

        # Decide actions for each agent
        for agent_idx in range(n_agents):
            if (
                torch.rand(1).item() < self.epsilon
            ):  # Exploration: Choose a random action
                available_actions = (
                    avail_actions[agent_idx].squeeze().nonzero(as_tuple=True)[0]
                )
                if len(available_actions) > 0:
                    chosen_actions[agent_idx] = available_actions[
                        torch.randint(0, len(available_actions), (1,))
                    ]
                else:
                    chosen_actions[agent_idx] = torch.tensor(
                        -1
                    )  # Handle case with no available actions
            else:  # Exploitation: Choose the best action
                chosen_actions[agent_idx] = (
                    masked_q_values[agent_idx].squeeze().max(0)[1]
                )

        return chosen_actions

    # ---- ---- ---- ---- ---- #
    # --- Partial Methods ---- #
    # ---- ---- ---- ---- ---- #

    decide_actions_epsilon_greedily = partialmethod(decide_actions, test_mode=False)
    decide_actions_greedily = partialmethod(decide_actions, test_mode=True)

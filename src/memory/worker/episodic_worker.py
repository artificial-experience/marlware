from functools import partial

import numpy as np

from src.memory.replay import EpisodeBatch


class EpisodeRunner:
    def __init__(self, conf, logger, env, env_info):
        self.conf = conf
        self.logger = logger
        self.batch_size = 1  # as this is a synchronous worker

        self.env = env
        self.env_info = env_info

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

    def setup(self, scheme, groups, preprocess, mac, device):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.infer_eps_greedy_actions(
                data=self.batch, rollout_timestep=self.t, env_timestep=self.t_env
            )

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            self.t_env += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.infer_eps_greedy_actions(
            data=self.batch, rollout_timestep=self.t, env_timestep=self.t_env
        )

        self.batch.update({"actions": actions}, ts=self.t)

        return self.batch

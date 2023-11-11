import random
from typing import Dict
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from src.cortex import MultiAgentCortex
from src.environ.starcraft import SC2Environ
from src.memory.buffer import GenericReplayMemory
from src.memory.buffer import initialize_memory
from src.memory.collector import SynchronousCollector
from src.registry import global_registry
from src.trainable import arch


class TrainableConstruct:
    """
    Abstraction class meant to delegate certain construct for optimization
    Based on conf and registry the construct is instantiated

    Args:
        :param [conf]: construct configuration OmegaConf

    Internal State:
        :param [trainable]: chosen trainable component to be used (e.g. BaseQMIX)
        :param [multi_agent_cortex]: controller to be used for multi-agent scenario
        :param [memory]: replay memory instance
        :param [environ]: environment instance
        :param [environ_info]: environment informations (e.g. number of actions)
        :param [optimizer]: optimizer used to backward pass grads through eval nets
        :param [params]: eval nets parameters
        :param [grad_clip]: gradient clip to prevent exploding gradients and divergence
        :param [trajectory_collector]: collector used to gather trajectories from environ

    """

    def __init__(self, conf: OmegaConf) -> None:
        self._conf = conf

        # internal attrs
        self._trainable: arch.TrainableComponent = None
        self._mac = None
        self._memory = None
        self._environ = None
        self._environ_info = None

        # optimization attrs
        self._optimizer = None
        self._params = []
        self._grad_clip = 10.0
        self._target_net_update_sched = 100
        self._accelerator = "cpu"

        # trajectory collector
        self._trajectory_collector = None

    def _integrate_trainable(
        self, construct: str, env_info: dict, gamma: float, seed: int
    ) -> arch.TrainableComponent:
        """check for registered constructs and integrate chosen one"""
        registered_constructs = global_registry.get_registered()
        is_registered = construct in registered_constructs
        assert is_registered is True, f"Construct {construct} is not registered"

        construct_hypernet_conf = self._conf.trainable.hypernetwork
        construct_mixer_conf = self._conf.trainable.mixer

        trainable = global_registry.get(construct)(
            construct_hypernet_conf, construct_mixer_conf
        )
        n_agents = env_info.get("n_agents", None)
        obs_dim = env_info.get("obs_shape", None)
        state_dim = env_info.get("state_shape", None)
        trainable.ensemble_construct(
            n_agents=n_agents,
            observation_dim=obs_dim,
            state_dim=state_dim,
            gamma=gamma,
            seed=seed,
        )
        return trainable

    def _integrate_multi_agent_cortex(
        self, model_conf: OmegaConf, exp_conf: OmegaConf, env_info: dict, seed: int
    ) -> MultiAgentCortex:
        """create multi-agent cortex for N agents"""
        n_agents = env_info.get("n_agents", None)
        n_actions = env_info.get("n_actions", None)
        obs_dim = env_info.get("obs_shape", None)
        mac = MultiAgentCortex(model_conf, exp_conf)
        mac.ensemble_cortex(n_agents, n_actions, obs_dim, seed=seed)
        return mac

    def _integrate_memory(
        self, memory_conf: OmegaConf, env_info: dict, seed: int
    ) -> GenericReplayMemory:
        """create instance of replay memory based on environ info and memory conf"""
        state_shape = env_info.get("state_shape", None)
        obs_shape = env_info.get("obs_shape", None)
        n_actions = env_info.get("n_actions", None)
        n_agents = env_info.get("n_agents", None)

        max_size = memory_conf.max_size
        batch_size = memory_conf.batch_size
        prioritized = memory_conf.prioritized

        prev_actions_field = "prev_actions"
        prev_actions_vals = np.zeros([max_size, n_agents, 1], dtype=np.int64)

        avail_actions_field = "avail_actions"
        avail_actions_vals = np.zeros([max_size, n_agents, n_actions], dtype=np.int64)

        states_field = "states"
        states_vals = np.zeros([max_size, state_shape], dtype=np.float32)

        next_states_field = "next_states"
        next_states_vals = np.zeros([max_size, state_shape], dtype=np.float32)

        extra_fields = (
            prev_actions_field,
            states_field,
            next_states_field,
            avail_actions_field,
        )
        extra_vals = (
            prev_actions_vals,
            states_vals,
            next_states_vals,
            avail_actions_vals,
        )

        memory = initialize_memory(
            obs_shape=(obs_shape,),
            n_actions=n_actions,
            n_agents=n_agents,
            max_size=max_size,
            batch_size=batch_size,
            prioritized=prioritized,
            extra_fields=extra_fields,
            extra_vals=extra_vals,
        )
        memory.ensemble_replay_memory(seed=seed)
        return memory

    def _integrate_environ(self, map_name: str) -> SC2Environ:
        """based on map_name create sc2 environ instance"""
        env_manager = SC2Environ(map_name)
        env, env_info = env_manager.create_env_instance()
        assert env is not None, "Environment cound not be created"
        return env, env_info

    def _integrate_collector(
        self,
        conf: OmegaConf,
        memory: GenericReplayMemory,
        environ: SC2Environ,
        env_info: dict,
    ) -> SynchronousCollector:
        collector = SynchronousCollector(conf)
        collector.ensemble_collector(memory, environ, env_info)
        return collector

    def _rnd_seed(self, *, seed: int = None):
        """set random seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def commit(
        self, environ_prefix: str, accelerator: str, *, seed: int = None
    ) -> None:
        """based on conf delegate construct object with given parameters"""
        self._rnd_seed(seed=seed)

        # ---- ---- ---- ---- ---- #
        # --- Integrate Environ -- #
        # ---- ---- ---- ---- ---- #

        self._environ, self._environ_info = self._integrate_environ(environ_prefix)

        # ---- ---- ---- ---- ---- #
        # -- Integrate Construct - #
        # ---- ---- ---- ---- ---- #

        gamma = self._conf.learner.training.gamma
        construct: str = self._conf.trainable.construct.impl
        self._trainable: arch.TrainableComponent = self._integrate_trainable(
            construct, self._environ_info, gamma, seed
        )

        # ---- ---- ---- ---- ---- #
        # --- Integrate Memory --- #
        # ---- ---- ---- ---- ---- #

        memory_conf = self._conf.buffer
        self._memory = self._integrate_memory(memory_conf, self._environ_info, seed)

        # ---- ---- ---- ---- ---- #
        # --- Integrate Cortex --- #
        # ---- ---- ---- ---- ---- #

        model_conf = self._conf.learner.model
        exp_conf = self._conf.learner.exploration
        self._mac = self._integrate_multi_agent_cortex(
            model_conf, exp_conf, self._environ_info, seed
        )

        # ---- ---- ---- ---- ---- #
        # --- Gather Params -- --- #
        # ---- ---- ---- ---- ---- #

        # eval mixer params
        construct_params = list(self._trainable.parameters())
        self._params.extend(construct_params)
        # eval drqn params
        cortex_params = list(self._mac.parameters())
        self._params.extend(cortex_params)

        # ---- ---- ---- ---- ---- #
        # --- Setup Optimizer  --- #
        # ---- ---- ---- ---- ---- #

        learning_rate = self._conf.learner.training.lr
        self._optimizer = torch.optim.RMSprop(params=self._params, lr=learning_rate)

        # ---- ---- ---- ---- ---- #
        # --- Setup Grad Clip ---- #
        # ---- ---- ---- ---- ---- #

        self._grad_clip = self._conf.learner.training.grad_clip

        # ---- ---- ---- ---- ---- #
        # --- Setup Target Update  #
        # ---- ---- ---- ---- ---- #

        target_network_update_schedule = (
            self._conf.learner.training.target_net_update_shedule
        )
        self._target_net_update_sched = target_network_update_schedule

        # ---- ---- ---- ---- ---- #
        # --- Setup Collector ---- #
        # ---- ---- ---- ---- ---- #

        collector_conf = self._conf.buffer
        self._trajectory_collector = self._integrate_collector(
            collector_conf, self._memory, self._environ, self._environ_info
        )

        # ---- ---- ---- ---- ---- #
        # --- Setup Accelerator -- #
        # ---- ---- ---- ---- ---- #

        self._accelerator = accelerator

    def _synchronize_target_nets(self):
        """synchronize target networks inside cortex and construct"""
        self._trainable.synchronize_target_net()
        self._mac.synchronize_target_net()

    def _get_avail_actions(self, n_agents: int) -> list:
        """return a list with available actions for each agent"""
        avail_actions = []
        for agent_id in range(n_agents):
            available_actions = self._environ.get_avail_agent_actions(agent_id)
            avail_actions.append(available_actions)

        return np.array(avail_actions, dtype=np.int64)

    def _evaluate(self, n_games: int = 40) -> np.ndarray:
        """evaluate construct on N games"""
        results = []
        for _ in range(n_games):
            self._environ.reset()
            terminated = False
            episode_return = 0
            prev_actions = torch.zeros((8, 1))

            while not terminated:
                observations = np.array(self._environ.get_obs(), dtype=np.float32)
                states = np.array(self._environ.get_state(), dtype=np.float32)
                avail_actions = self._get_avail_actions(8)

                actions: np.ndarray = self._mac.compute_greedy_actions(
                    observations, prev_actions, avail_actions, 0
                )

                reward, terminated, _ = self._environ.step(actions)

                episode_return += reward
                prev_actions = actions

            results.append(episode_return)

        mean_score = np.mean(results)
        print("EVAL MEAN SCORE: ", mean_score)
        return mean_score

    def _move_batch_to_tensors(self, batch: list) -> list:
        """Move numpy arrays to torch tensors"""
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            prev_actions,
            states,
            next_states,
            avail_actions,
        ) = batch

        # Convert each numpy array in the batch to a torch tensor and move it to the specified device
        observations = torch.tensor(observations, dtype=torch.float32).to(
            self._accelerator
        )
        actions = torch.tensor(actions, dtype=torch.int64).to(self._accelerator)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self._accelerator)
        next_observations = torch.tensor(next_observations, dtype=torch.float32).to(
            self._accelerator
        )
        dones = torch.tensor(dones, dtype=torch.int8).to(self._accelerator)
        prev_actions = torch.tensor(prev_actions, dtype=torch.int64).to(
            self._accelerator
        )
        states = torch.tensor(states, dtype=torch.float32).to(self._accelerator)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(
            self._accelerator
        )
        avail_actions = torch.tensor(avail_actions, dtype=torch.int64).to(
            self._accelerator
        )

        # Return the tensors as a list
        return [
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            prev_actions,
            states,
            next_states,
            avail_actions,
        ]

    def optimize(
        self,
        n_rollouts: int,
        eval_schedule: int,
        checkpoint_freq: int,
        eval_n_games: int,
    ) -> np.ndarray:
        """optimize construct within N rollouts"""

        for rollout in range(n_rollouts):
            # synchronize target nets with online updates
            if rollout % self._target_net_update_sched == 0:
                self._synchronize_target_nets()

            # evaluate performance on n_games
            if rollout % eval_schedule == 0:
                self._evaluate(eval_n_games)

            # if memory ready optimize network
            if self._trajectory_collector.memory_ready():
                # zero optimizer
                self._optimizer.zero_grad()

                batch = self._trajectory_collector.sample_batch()
                t_batch = self._move_batch_to_tensors(batch)
                # unpack batch
                (
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    dones,
                    prev_actions,
                    states,
                    next_states,
                    avail_actions,
                ) = t_batch

                # prepare feed for networks
                eval_net_feed = {
                    "observations": observations,
                    "actions": prev_actions,
                    "avail_actions": avail_actions,
                }

                target_net_feed = {
                    "observations": next_observations,
                    "actions": actions,
                    "avail_actions": avail_actions,
                }

                # n_agents X batch_size X n_q_values
                eval_net_q_estimates = self._mac.estimate_eval_q_vals(
                    feed=eval_net_feed
                )
                target_net_q_estimates = self._mac.estimate_target_q_vals(
                    feed=target_net_feed
                )

                construct_feed = {
                    "prev_actions": prev_actions,
                    "actions": actions,
                    "states": states,
                    "next_states": next_states,
                    "avail_actions": avail_actions,
                    "rewards": rewards,
                    "dones": dones,
                }

                construct_loss = self._trainable.calculate_loss(
                    feed=construct_feed,
                    eval_q_vals=eval_net_q_estimates,
                    target_q_vals=target_net_q_estimates,
                )
                construct_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._params, self._grad_clip
                )
                self._optimizer.step()

                print("Current Rollout: ", rollout)
                print("Current Loss: ", construct_loss)

            # use multi agent cortex to collect rollouts
            self._trajectory_collector.roll_environ_and_collect_trajectory(
                mac=self._mac
            )

    def save_models(self) -> bool:
        """save all models"""
        pass

    def load_models(self, path_to_models: str) -> bool:
        """load all models given path"""
        pass
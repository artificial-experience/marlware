import random
from itertools import chain
from logging import Logger
from typing import List
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torchviz import make_dot

from src import trainable
from src.cortex import MultiAgentCortex
from src.environ.starcraft import SC2Environ
from src.evaluator import CoreEvaluator
from src.memory.replay import ReplayBuffer
from src.memory.worker import EpisodeRunner
from src.registry import trainable_global_registry
from src.transforms import OneHotTransform
from src.util.constants import AttrKey


class CoreTuner:
    """
    Abstraction class meant to delegate certain trainable for optimization
    Based on conf and registry the trainable is instantiated
    Prototype for other tuner classes

    Args:
        :param [conf]: trainable configuration OmegaConf

    Internal State:
        :param [trainable]: chosen trainable component to be used (e.g. QmixCore)
        :param [multi_agent_cortex]: controller to be used for multi-agent scenario
        :param [memory]: replay memory instance
        :param [environ]: environment instance
        :param [environ_info]: environment informations (e.g. number of actions)
        :param [optimizer]: optimizer used to backward pass grads through eval nets
        :param [params]: eval nets parameters
        :param [grad_clip]: gradient clip to prevent exploding gradients and divergence
        :param [trajectory_worker]: worker used to gather trajectories from environ

    """

    def __init__(self, conf: OmegaConf) -> None:
        self._conf = conf

        # internal attrs
        self._trainable: trainable.Trainable = None
        self._mac = None
        self._memory = None
        self._environ = None
        self._environ_info = None
        self._trace_logger = None

        # optimization attrs
        self._optimizer = None
        self._params = []
        self._grad_clip = 10.0
        self._target_net_update_sched = 100
        self._accelerator = "cpu"

        # trajectory worker
        self._trajectory_worker = None

        # evaluator
        self._evaluator = None

    def _integrate_trainable(
        self, trainable: str, env_info: dict, gamma: float, seed: int
    ) -> trainable.Trainable:
        """check for registered trainable and integrate chosen one"""
        registered_trainables = trainable_global_registry.get_registered()
        is_registered = trainable in registered_trainables
        assert is_registered is True, f"Trainable {trainable} is not registered"

        trainable_hypernet_conf = self._conf.trainable.hypernetwork
        trainable_mixer_conf = self._conf.trainable.mixer

        trainable = trainable_global_registry.get(trainable)(
            trainable_hypernet_conf, trainable_mixer_conf
        )
        n_agents = env_info.get("n_agents", None)
        obs_dim = env_info.get("obs_shape", None)
        state_dim = env_info.get("state_shape", None)
        trainable.ensemble_trainable(
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
        self,
        memory_conf: OmegaConf,
        scheme: dict,
        groups: dict,
        max_seq_length: int,
        preprocess: dict,
        accelerator: str,
        seed: int,
    ) -> ReplayBuffer:
        """create instance of replay memory based on environ info and memory conf"""
        batch_size = memory_conf.get("batch_size", None)
        memory = ReplayBuffer(
            scheme=scheme,
            groups=groups,
            buffer_size=batch_size,
            max_seq_length=max_seq_length + 1,
            preprocess=preprocess,
            device=accelerator,
        )
        return memory

    def _integrate_worker(
        self,
        conf: OmegaConf,
        logger: Logger,
        env: SC2Environ,
        env_info: dict,
    ) -> EpisodeRunner:
        """create worker instance to be used for interaction with env"""
        worker = EpisodeRunner(conf, logger, env, env_info)
        return worker

    def _integrate_environ(self, map_name: str) -> SC2Environ:
        """based on map_name create sc2 environ instance"""
        env_manager = SC2Environ(map_name)
        env, env_info = env_manager.create_env_instance()
        assert env is not None, "Environment cound not be created"
        return env, env_info

    def _integrate_evaluator(
        self, env: SC2Environ, env_info: dict, cortex: MultiAgentCortex, logger: Logger
    ) -> CoreEvaluator:
        """create evaluator instance"""
        evaluator = CoreEvaluator()
        evaluator.ensemble_evaluator(
            env=env, env_info=env_info, cortex=cortex, logger=logger
        )
        return evaluator

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def commit(
        self,
        environ_prefix: str,
        accelerator: str,
        logger: Logger,
        *,
        seed: Optional[int] = None,
    ) -> None:
        """based on conf delegate tuner object with given parameters"""
        self._rnd_seed(seed=seed)

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Accelerator
        # ---- ---- ---- ---- ---- #

        self._accelerator = accelerator

        # ---- ---- ---- ---- ---- #
        # @ ->  - Setup Logger
        # ---- ---- ---- ---- ---- #

        self._trace_logger = logger

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Environ
        # ---- ---- ---- ---- ---- #

        self._environ, self._environ_info = self._integrate_environ(environ_prefix)

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Trainable
        # ---- ---- ---- ---- ---- #

        gamma = self._conf.learner.training.gamma
        trainable: str = self._conf.trainable.construct.impl
        self._trainable: trainable.Trainable = self._integrate_trainable(
            trainable, self._environ_info, gamma, seed
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Cortex
        # ---- ---- ---- ---- ---- #

        model_conf = self._conf.learner.model
        exp_conf = self._conf.learner.exploration
        self._mac = self._integrate_multi_agent_cortex(
            model_conf, exp_conf, self._environ_info, seed
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Gather Params
        # ---- ---- ---- ---- ---- #

        # eval mixer params
        trainable_params = list(self._trainable.parameters())
        self._params.extend(trainable_params)

        # eval drqn params
        cortex_params = list(self._mac.parameters())
        self._params.extend(cortex_params)

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Optimizer
        # ---- ---- ---- ---- ---- #

        learning_rate = self._conf.learner.training.lr
        # TODO: move alpha and eps to config file
        self._optimizer = torch.optim.RMSprop(
            params=self._params, lr=learning_rate, alpha=0.99, eps=1e-5
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Grad Clip
        # ---- ---- ---- ---- ---- #

        self._grad_clip = self._conf.learner.training.grad_clip

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Target Update
        # ---- ---- ---- ---- ---- #

        target_network_update_schedule = (
            self._conf.learner.training.target_net_update_shedule
        )
        self._target_net_update_sched = target_network_update_schedule

        # ---- ---- ---- ---- ---- #
        # @ -> Prepare Blueprint
        # ---- ---- ---- ---- ---- #

        data_attr = AttrKey.data
        env_attr = AttrKey.env

        state_shape = self._environ_info[env_attr._STATE_SHAPE.value]
        obs_shape = self._environ_info[env_attr._OBS_SHAPE.value]
        n_agents = self._environ_info[env_attr._N_AGENTS.value]
        n_actions = self._environ_info[env_attr._N_ACTIONS.value]
        max_seq_length = self._environ_info[env_attr._EP_LIMIT.value]

        # create scheme blueprint
        scheme = {
            data_attr._STATE.value: {data_attr._VALUE_SHAPE.value: state_shape},
            data_attr._OBS.value: {
                data_attr._VALUE_SHAPE.value: obs_shape,
                data_attr._GROUP.value: data_attr._AGENT_GROUP.value,
            },
            data_attr._ACTIONS.value: {
                data_attr._VALUE_SHAPE.value: (1,),
                data_attr._GROUP.value: data_attr._AGENT_GROUP.value,
                data_attr._DTYPE.value: torch.int64,
            },
            data_attr._AVAIL_ACTIONS.value: {
                data_attr._VALUE_SHAPE.value: (n_actions,),
                data_attr._GROUP.value: data_attr._AGENT_GROUP.value,
                data_attr._DTYPE.value: torch.int64,
            },
            data_attr._PROBS.value: {
                data_attr._VALUE_SHAPE.value: (n_actions,),
                data_attr._GROUP.value: data_attr._AGENT_GROUP.value,
                data_attr._DTYPE.value: torch.float32,
            },
            data_attr._REWARD.value: {data_attr._VALUE_SHAPE.value: (1,)},
            data_attr._TERMINATED.value: {
                data_attr._VALUE_SHAPE.value: (1,),
                data_attr._DTYPE.value: torch.int64,
            },
        }

        # create groups blueprint
        groups = {data_attr._AGENT_GROUP.value: n_agents}

        # prepare preprocessing blueprint
        preprocess = {
            data_attr._ACTIONS.value: (
                data_attr._ACTIONS_ONEHOT_TRANSFORM.value,
                [OneHotTransform(out_dim=n_actions)],
            )
        }

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Memory
        # ---- ---- ---- ---- ---- #

        memory_conf = self._conf.buffer
        self._memory = self._integrate_memory(
            memory_conf,
            scheme,
            groups,
            max_seq_length,
            preprocess,
            self._accelerator,
            seed,
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Worker
        # ---- ---- ---- ---- ---- #

        worker_conf = self._conf.buffer
        self._trajectory_worker = self._integrate_worker(
            worker_conf, self._trace_logger, self._environ, self._environ_info
        )
        self._trajectory_worker.setup(
            scheme, groups, preprocess, self._mac, self._accelerator
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Evaluator
        # ---- ---- ---- ---- ---- #

        self._evaluator = self._integrate_evaluator(
            self._environ, self._environ_info, self._mac, self._trace_logger
        )
        self._evaluator.setup(scheme, groups, preprocess, self._accelerator)

    def _synchronize_target_nets(self):
        """synchronize target networks inside cortex and trainable"""
        self._trainable.synchronize_target_net()
        self._mac.synchronize_target_net()

    # --------------------------------------------------
    # @ -> Tuner optimization mechanizm
    # --------------------------------------------------

    def optimize(
        self,
        n_rollouts: int,
        eval_schedule: int,
        checkpoint_freq: int,
        eval_n_games: int,
        display_freq: int = 100,
        timesteps_max: int = 10_000_000,
        batch_size: int = 32,
    ) -> np.ndarray:
        """optimize trainable within N rollouts"""
        rollout = 0
        while self._trajectory_worker.t_env <= timesteps_max:
            # ---- ---- ---- ---- ---- #
            # @ -> Synchronize Nets
            # ---- ---- ---- ---- ---- #

            if rollout % self._target_net_update_sched == 0:
                self._synchronize_target_nets()

            # ---- ---- ---- ---- ---- #
            # @ -> Evaluate Performance
            # ---- ---- ---- ---- ---- #

            if rollout % eval_schedule == 0:
                self._evaluator.evaluate(rollout=rollout, n_games=eval_n_games)

            # ---- ---- ---- ---- ---- #
            # @ -> Gather Rollouts
            # ---- ---- ---- ---- ---- #

            # Run for a whole episode at a time
            with torch.no_grad():
                episode_batch = self._trajectory_worker.run()
                self._memory.insert_episode_batch(episode_batch)

            if self._memory.can_sample(batch_size):
                episode_sample = self._memory.sample(batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != self._accelerator:
                    episode_sample.to(self._accelerator)

                # ---- ---- ---- ---- ---- #
                # @ -> Calculate Q-Vals
                # ---- ---- ---- ---- ---- #

                # initialize hidden states of network
                self._mac.init_hidden(batch_size=batch_size)

                timewise_eval_estimates = []
                timewise_target_estimates = []

                max_seq_length = episode_sample.max_seq_length
                for seq_t in range(max_seq_length):
                    # timewise slices of episodes
                    episode_time_slice = episode_sample[:, seq_t]

                    eval_net_q_estimates = self._mac.estimate_eval_q_vals(
                        feed=episode_time_slice
                    )
                    target_net_q_estimates = self._mac.estimate_target_q_vals(
                        feed=episode_time_slice
                    )

                    # append timewise list with agent q estimates
                    timewise_eval_estimates.append(eval_net_q_estimates)
                    timewise_target_estimates.append(target_net_q_estimates)

                # stack estimates timewise
                t_timewise_eval_estimates = torch.stack(timewise_eval_estimates, dim=1)
                t_timewise_target_estimates = torch.stack(
                    timewise_target_estimates, dim=1
                )

                # ---- ---- ---- ---- ---- #
                # @ -> Calculate Loss
                # ---- ---- ---- ---- ---- #

                # eval does not need last estimate
                t_timewise_eval_estimates = t_timewise_eval_estimates[:, :-1]

                # target does not need first estimate
                t_timewise_target_estimates = t_timewise_target_estimates[:, 1:]

                trainable_loss = self._trainable.calculate_loss(
                    feed=episode_sample,
                    eval_q_vals=t_timewise_eval_estimates,
                    target_q_vals=t_timewise_target_estimates,
                )
                self._optimizer.zero_grad()
                trainable_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._params, self._grad_clip
                )
                self._optimizer.step()

                self._trace_logger.log_stat("trainable_loss", trainable_loss, rollout)
                self._trace_logger.log_stat("gradient_norm", grad_norm, rollout)

            # ---- ---- ---- ---- ---- #
            # @ -> Log Stats
            # ---- ---- ---- ---- ---- #

            if rollout % display_freq == 0:
                self._trace_logger.display_recent_stats()

            # update episode counter ( works for synchronous training )
            rollout += 1

    # --------------------------------------------------
    # @ -> Methods for saving models and debugging
    # --------------------------------------------------

    def draw_computational_graph(self, dummy_episode_sample) -> None:
        """Draw computational graph using torchviz

        Args:
        dummy_episode_sample: A representative sample of the episode data.
        """

        # Ensure the dummy input is on the same device as the model
        if dummy_episode_sample.device != self._accelerator:
            dummy_episode_sample.to(self._accelerator)

        # Initialize hidden states of network
        self._mac.init_hidden(batch_size=dummy_episode_sample.batch_size)

        timewise_eval_estimates = []
        timewise_target_estimates = []
        max_seq_length = dummy_episode_sample.max_seq_length

        for seq_t in range(max_seq_length):
            # Timewise slices of episodes
            episode_time_slice = dummy_episode_sample[:, seq_t]

            eval_net_q_estimates = self._mac.estimate_eval_q_vals(
                feed=episode_time_slice
            )
            target_net_q_estimates = self._mac.estimate_target_q_vals(
                feed=episode_time_slice
            )

            # Append timewise list with agent q estimates
            timewise_eval_estimates.append(eval_net_q_estimates)
            timewise_target_estimates.append(target_net_q_estimates)

        # Stack estimates timewise
        t_timewise_eval_estimates = torch.stack(timewise_eval_estimates, dim=1)
        t_timewise_target_estimates = torch.stack(timewise_target_estimates, dim=1)

        # Truncate to only filled timesteps
        t_timewise_eval_estimates = t_timewise_eval_estimates[:, :-1]
        t_timewise_target_estimates = t_timewise_target_estimates[:, 1:]

        # Calculate loss
        trainable_loss = self._trainable.calculate_loss(
            feed=dummy_episode_sample,
            eval_q_vals=t_timewise_eval_estimates,
            target_q_vals=t_timewise_target_estimates,
        )

        # Manually assign names to parameters and collect them in a dictionary
        all_parameters = chain(self._mac.parameters(), self._trainable.parameters())
        named_parameters = {f"param_{i}": p for i, p in enumerate(all_parameters)}

        # Generate the graph from the loss
        graph = make_dot(
            trainable_loss, params=named_parameters, show_attrs=True, show_saved=True
        )

        # Render the graph to a file
        graph.render("computational_graph", format="png", cleanup=True)

    def save_models(self) -> bool:
        """save all models"""
        pass

    def load_models(self, path_to_models: str) -> bool:
        """load all models given path"""
        pass

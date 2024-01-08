import os
import random
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from src import trainable
from src.abstract import ProtoTuner
from src.cortex import RecQCortex
from src.environ.starcraft import SC2Environ
from src.evaluator import CoreEvaluator
from src.memory.cluster import MemoryCluster
from src.registry import trainable_global_registry
from src.transforms import OneHotTransform
from src.util import constants
from src.util.constants import AttrKey
from src.worker import InteractionWorker


class ProtoTuner(ProtoTuner):
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
        :param [interaction_worker]: worker used to gather trajectories from environ

    """

    def __init__(self, conf: OmegaConf) -> None:
        self._conf = conf

        # internal attrs
        self._trainable: trainable.Trainable = None
        self._mac = None
        self._memory_cluster = None
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
        self._interaction_worker = None

        # evaluator
        self._evaluator = None

        # run identifier
        self._run_identifier = None

        # ray sutff
        self._ray_map = defaultdict(int)

        # num workers
        self._num_worker_handlers = 1

    # --------------------------------------------------
    # @ -> Methods for component integration
    # --------------------------------------------------

    def _integrate_trainable(
        self,
        trainable: str,
        n_agents: int,
        obs_shape: tuple,
        state_shape: tuple,
        gamma: float,
        seed: int,
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
        trainable.ensemble_trainable(
            n_agents=n_agents,
            observation_dim=obs_shape,
            state_dim=state_shape,
            gamma=gamma,
            seed=seed,
        )
        return trainable

    def _integrate_multi_agent_cortex(
        self,
        model_conf: OmegaConf,
        exp_conf: OmegaConf,
        n_agents: int,
        n_actions: int,
        obs_dim: tuple,
        seed: int,
    ) -> RecQCortex:
        """create multi-agent cortex for N agents"""
        mac = RecQCortex(model_conf, exp_conf)
        mac.ensemble_cortex(n_agents, n_actions, obs_dim, seed=seed)
        return mac

    def _integrate_memory(
        self,
        mem_size: int,
        sampling_method: str,
        seed: int,
    ) -> MemoryCluster:
        """create instance of replay memory based on environ info and memory conf"""
        memory = MemoryCluster(mem_size)
        memory.ensemble_memory_cluster(sampling_method=sampling_method, seed=seed)
        return memory

    def _integrate_worker(
        self,
        env: List[ray.ObjectRef],
        cortex: ray.ObjectRef,
        memory_blueprint: dict,
        accelerator: str,
        num_workers: int,
        replay_save_path: Path,
    ) -> InteractionWorker:
        """create worker instance to be used for interaction with env"""
        worker_handlers = []
        for worker_id in range(num_workers):
            worker_replay_save_dir = replay_save_path / f"worker-{worker_id}"
            worker = InteractionWorker.remote()
            worker.ensemble_interaction_worker.remote(
                env=env[worker_id],
                cortex=cortex,
                memory_blueprint=memory_blueprint,
                device=accelerator,
                replay_save_path=worker_replay_save_dir,
            )
            worker_handlers.append(worker)
        return worker_handlers

    def _integrate_environ(
        self, env_conf: str, num_envs: int, seed: list
    ) -> SC2Environ:
        """based on env conf create sc2 environ instance"""
        env_manager = SC2Environ(env_conf)
        env_list = []
        env_info_list = []
        for env_idx in range(num_envs):
            env, env_info = env_manager.create_env_instance(seed=seed[env_idx])
            assert env is not None, "Environment cound not be created"
            assert env_info is not None, "Environment info cound not be created"

            env_list.append(env)
            env_info_list.append(env_info)

        return env_list, env_info_list

    def _integrate_evaluator(
        self,
        worker: InteractionWorker,
    ) -> CoreEvaluator:
        """create evaluator instance"""
        evaluator = CoreEvaluator.remote(worker)
        evaluator.ensemble_evaluator.remote()
        return evaluator

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    # --------------------------------------------------
    # @ -> Commit configuration for tuner instance
    # --------------------------------------------------

    def commit(
        self,
        env_conf: str,
        accelerator: str,
        logger: Logger,
        run_id: str,
        *,
        num_workers: Optional[int] = 4,
        seed: Optional[list] = None,
    ) -> None:
        """based on conf delegate tuner object with given parameters"""
        self._rnd_seed(seed=seed[0])

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Num Workers
        # ---- ---- ---- ---- ---- #

        self._num_worker_handlers = num_workers

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Run Idenfifier
        # ---- ---- ---- ---- ---- #

        self._run_identifier = run_id

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Accelerator
        # ---- ---- ---- ---- ---- #

        self._accelerator = accelerator

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Logger
        # ---- ---- ---- ---- ---- #

        self._trace_logger = logger

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Environ
        # ---- ---- ---- ---- ---- #

        # returns list of envs and infos
        envs_list, envs_info_list = self._integrate_environ(env_conf, num_workers, seed)
        self._environ = envs_list
        self._environ_info = envs_info_list[0]

        # ---- ---- ---- ---- ---- #
        # @ -> Access Attr Keys
        # ---- ---- ---- ---- ---- #

        data_attr = AttrKey.data
        env_attr = AttrKey.env
        tuner_attr = AttrKey.tuner

        n_agents = self._environ_info[env_attr._N_AGENTS.value]
        n_actions = self._environ_info[env_attr._N_ACTIONS.value]
        obs_shape = self._environ_info[env_attr._OBS_SHAPE.value]
        state_shape = self._environ_info[env_attr._STATE_SHAPE.value]

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Trainable
        # ---- ---- ---- ---- ---- #

        gamma = self._conf.learner.training.gamma
        trainable: str = self._conf.trainable.construct.impl
        self._trainable: trainable.Trainable = self._integrate_trainable(
            trainable, n_agents, obs_shape, state_shape, gamma, seed[0]
        )
        self._trainable.move_to_device(device=self._accelerator)

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Cortex
        # ---- ---- ---- ---- ---- #

        model_conf = self._conf.learner.model
        exp_conf = self._conf.learner.exploration
        self._mac = self._integrate_multi_agent_cortex(
            model_conf, exp_conf, n_agents, n_actions, obs_shape, seed[0]
        )
        self._mac.move_to_device(device=self._accelerator)

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
        self._optimizer = torch.optim.Adam(params=self._params, lr=learning_rate)

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
        transforms = {
            data_attr._ACTIONS.value: (
                data_attr._ACTIONS_ONEHOT_TRANSFORM.value,
                [OneHotTransform(out_dim=n_actions)],
            )
        }

        memory_blueprint = {
            data_attr._SCHEME.value: scheme,
            data_attr._GROUP.value: groups,
            data_attr._MAX_SEQ_LEN.value: max_seq_length,
            data_attr._TRANSFORMS.value: transforms,
        }

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Memory
        # ---- ---- ---- ---- ---- #

        mem_size = self._conf.buffer[tuner_attr._MEM_SIZE.value]
        sampling_method = self._conf.buffer[tuner_attr._SAMPLING_METHOD.value]
        self._memory_cluster = self._integrate_memory(
            mem_size,
            sampling_method,
            seed[0],
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Ray Store and Actors
        # ---- ---- ---- ---- ---- #

        self.put_environ_into_object_store()
        self.put_cortex_into_object_store()

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Worker
        # ---- ---- ---- ---- ---- #

        env_ref = self._ray_map["env"]
        mac_ref = self._ray_map["mac"]
        replay_save_path: Path = constants.REPLAY_DIR / self._run_identifier
        self._interaction_worker = self._integrate_worker(
            env_ref,
            mac_ref,
            memory_blueprint,
            self._accelerator,
            num_workers,
            replay_save_path,
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Evaluator
        # ---- ---- ---- ---- ---- #

        self._evaluator = self._integrate_evaluator(
            self._interaction_worker[0],
        )

    def _synchronize_target_nets(self):
        """synchronize target networks inside cortex and trainable"""
        self._trainable.synchronize_target_net()
        self._mac.synchronize_target_net()

    # --------------------------------------------------
    # @ -> Methods for interaction with ray engine
    # --------------------------------------------------

    def put_environ_into_object_store(self) -> None:
        """put environments into object store in ray"""
        env_refs = []
        for env in self._environ:
            env_ref = ray.put(env)
            env_refs.append(env_ref)
        self._ray_map["env"] = env_refs

    def put_cortex_into_object_store(self) -> None:
        """put cortex component into ray object store"""
        mac_ref = ray.put(self._mac)
        self._ray_map["mac"] = mac_ref

    def fetch_total_elapsed_timesteps(self) -> int:
        """get summ of all timesteps passed from workers"""
        timesteps = []
        for worker in self._interaction_worker:
            ts_future = worker.fetch_elapsed_timesteps.remote()
            worker_ts = ray.get(ts_future)
            timesteps.append(worker_ts)
        return sum(timesteps)

    def update_cortex_in_actor_handlers(self, cortex_ref: ray.ObjectRef) -> None:
        """update actors of a new reference to cortex"""
        assert self._interaction_worker is not None, "Worker can not be None"
        for worker in self._interaction_worker:
            worker.update_cortex_object.remote(cortex_ref)

    # --------------------------------------------------
    # @ -> Methods for saving and loading models
    # --------------------------------------------------

    def save_models(self, model_identifier: str) -> bool:
        """save all models"""
        save_directory = constants.MODEL_SAVE_DIR / self._run_identifier

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self._mac.save_models(save_directory, model_identifier)
        self._trainable.save_models(save_directory, model_identifier)

    def load_models(self, path_to_models: str) -> bool:
        """load all models given path"""
        pass

    # --------------------------------------------------
    # @ -> Methods for parsing and logging metrics
    # --------------------------------------------------

    def log_metrics(self, metrics: dict) -> None:
        """log metrics using logger"""
        mean_performance = metrics["mean_performance"]
        mean_scores = metrics["mean_scores"]
        mean_won_battles = metrics["mean_won_battles"]
        best_score = metrics["best_score"]
        highest_battle_win_score = metrics["highest_battle_win_score"]
        rollout = metrics["rollout"]

        if not mean_performance:
            return

        self._trace_logger.log_stat("eval_score_mean", mean_scores, rollout)

        # Calculate statistics
        eval_running_mean = np.mean(mean_performance)
        eval_score_std = np.std(mean_performance)

        # Log running mean and standard deviation
        self._trace_logger.log_stat(
            "eval_score_running_mean", eval_running_mean, rollout
        )
        self._trace_logger.log_stat("eval_score_std", eval_score_std, rollout)
        self._trace_logger.log_stat("eval_won_battles_mean", mean_won_battles, rollout)
        self._trace_logger.log_stat(
            "eval_most_won_battles", highest_battle_win_score, rollout
        )
        self._trace_logger.log_stat("eval_mean_higest_score", best_score, rollout)

        # Calculate and log the variation between the most recent two evaluations, if available
        if len(mean_performance) >= 2:
            # Calculate the variation between the mean scores of the last two evaluations
            recent_mean_scores = [np.mean(sublist) for sublist in mean_performance[-2:]]
            eval_score_var = np.abs(recent_mean_scores[-1] - recent_mean_scores[-2])
            self._trace_logger.log_stat("eval_score_var", eval_score_var, rollout)

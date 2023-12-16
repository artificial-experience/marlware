import logging
import os
import random
from logging import Logger
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from src import trainable
from src.abstract import ProtoTuner
from src.cortex import RecQCortex
from src.environ.starcraft import SC2Environ
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
        accelerator: str,
        seed: int,
    ) -> MemoryCluster:
        """create instance of replay memory based on environ info and memory conf"""
        memory = MemoryCluster()
        return memory

    def _integrate_worker(
        self,
        env: SC2Environ,
        cortex: RecQCortex,
        memory_blueprint: dict,
        logger: Logger,
        accelerator: str,
    ) -> InteractionWorker:
        """create worker instance to be used for interaction with env"""
        worker = InteractionWorker()
        worker.ensemble_interaction_worker(
            env=env,
            cortex=cortex,
            memory_blueprint=memory_blueprint,
            logger=logger,
            device=accelerator,
        )
        return worker

    def _integrate_environ(self, map_name: str) -> SC2Environ:
        """based on map_name create sc2 environ instance"""
        env_manager = SC2Environ(map_name)
        env, env_info = env_manager.create_env_instance()
        assert env is not None, "Environment cound not be created"
        return env, env_info

    # TODO: 1. fill this method
    def _integrate_evaluator(self):
        """create evaluator instance"""
        pass

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
        environ_prefix: str,
        accelerator: str,
        logger: Logger,
        run_id: str,
        *,
        seed: Optional[int] = None,
    ) -> None:
        """based on conf delegate tuner object with given parameters"""
        self._rnd_seed(seed=seed)

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

        self._environ, self._environ_info = self._integrate_environ(environ_prefix)

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
            trainable, n_agents, obs_shape, state_shape, gamma, seed
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Cortex
        # ---- ---- ---- ---- ---- #

        model_conf = self._conf.learner.model
        exp_conf = self._conf.learner.exploration
        self._mac = self._integrate_multi_agent_cortex(
            model_conf, exp_conf, n_agents, n_actions, obs_shape, seed
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
        # TODO: 2. move alpha and eps to config file
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

        memory_blueprint = {
            data_attr._SCHEME.value: scheme,
            data_attr._GROUP.value: groups,
            data_attr._MAX_SEQ_LEN.value: max_seq_length,
            data_attr._TRANSFORMS.value: preprocess,
        }

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Memory
        # ---- ---- ---- ---- ---- #

        mem_size = self._conf.buffer[tuner_attr._MEM_SIZE.value]
        self._memory_cluster = self._integrate_memory(
            self._accelerator,
            seed,
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Setup Worker
        # ---- ---- ---- ---- ---- #

        self._interaction_worker = self._integrate_worker(
            self._environ,
            self._mac,
            memory_blueprint,
            self._trace_logger,
            self._accelerator,
        )

        # ---- ---- ---- ---- ---- #
        # @ -> Integrate Evaluator
        # ---- ---- ---- ---- ---- #

        # TODO: 1. implement evaluator stuff
        self._evaluator = None

    def _synchronize_target_nets(self):
        """synchronize target networks inside cortex and trainable"""
        self._trainable.synchronize_target_net()
        self._mac.synchronize_target_net()

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

from logging import Logger
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from .proto import ProtoTuner


class SyncTuner(ProtoTuner):
    """
    Synchronous tuner class meant to optimize trainable w.r.t objective function

    Args:
        :param [conf]: trainable configuration OmegaConf

    Derived State:
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
        super().__init__(conf)

    def commit(
        self,
        environ_prefix: str,
        accelerator: str,
        logger: Logger,
        *,
        seed: Optional[int] = None,
    ) -> None:
        """based on conf delegate tuner object with given parameters"""
        super().commit(environ_prefix, accelerator, logger, seed=seed)

    # --------------------------------------------------
    # @ -> Tuner optimization mechanizm
    # --------------------------------------------------

    def optimize(
        self,
        n_timesteps: int,
        batch_size: int,
        eval_schedule: int,
        checkpoint_freq: int,
        eval_n_games: int,
        display_freq: int,
    ) -> np.ndarray:
        """optimize trainable within N rollouts"""
        rollout = 0
        while self._trajectory_worker.t_env <= n_timesteps:
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

                self._trace_logger.log_stat(
                    "timesteps_passed", self._trajectory_worker.t_env, rollout
                )
                self._trace_logger.log_stat("trainable_loss", trainable_loss, rollout)
                self._trace_logger.log_stat("gradient_norm", grad_norm, rollout)

            # ---- ---- ---- ---- ---- #
            # @ -> Log Stats
            # ---- ---- ---- ---- ---- #

            if rollout % display_freq == 0:
                self._trace_logger.display_recent_stats()

            # update episode counter ( works for synchronous training )
            rollout += 1

        # close environment once the work is done
        self._environ.close()

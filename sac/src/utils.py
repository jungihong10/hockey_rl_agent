import os
import yaml

import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import gymnasium as gym

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

import hockey.hockey_env as h_env


def load_config(config_path):
    """Loads hyperparameters from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def custom_evaluate_policy(env, model, n_eval_episodes):
    rew_stats = []
    touch_stats = []
    won_stats = []
    lost_stats = []

    ob, _ = env.reset()
    for episode_counter in range(n_eval_episodes):
        total_reward = 0
        

        touch = 0
        won = 0
        lost = 0

        for step in range(env.unwrapped.max_timesteps):

            a1, _ = model.predict(ob, deterministic=True)  # Predict action for agent

            ob, reward, done, _, info = env.step(a1)

            total_reward += reward
            touch += int(info['reward_touch_puck'] > 0)

            if done:
                won = int(env.unwrapped.winner == 1)
                lost = int(env.unwrapped.winner == -1)
                ob, _ = env.reset()
                break

        rew_stats.append(total_reward)
        touch_stats.append(touch)
        won_stats.append(won)
        lost_stats.append(lost)

    return np.mean(rew_stats), np.mean(touch_stats), np.mean(won_stats), np.mean(lost_stats)
    


class CustomCheckpointCallback(EvalCallback):
    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
        ):

        if log_path is not None:
            self.custom_log_path  = os.path.join(log_path, "custom_evaluations")

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn
        )


        self.rew_stats: list[float] = []
        self.touch_stats: list[float] = []
        self.won_stats: list[float] = []
        self.lost_stats: list[float] = []

        self.rew_stats_weak_opp: list[float] = []
        self.touch_stats_weak_opp: list[float] = []
        self.won_stats_weak_opp: list[float] = []
        self.lost_stats_weak_opp: list[float] = []

    
    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            env = self.eval_env.envs[0]
            strong_opp_env = env.update_opponent_model(h_env.BasicOpponent(weak=False))

            avg_rew_stats, avg_touch_stats, avg_won_stats, avg_lost_stats = custom_evaluate_policy(
                strong_opp_env,
                self.model,
                n_eval_episodes=self.n_eval_episodes
            )

            self.rew_stats.append(avg_rew_stats)
            self.touch_stats.append(avg_touch_stats)
            self.won_stats.append(avg_won_stats)
            self.lost_stats.append(avg_lost_stats)

            print("strong opponent:")
            print(f"Avg Reward: {self.rew_stats[-1]:.2f}, Touch Rate: {self.touch_stats[-1]:.2f}, "
                f"Win Rate: {self.won_stats[-1]:.2f}, Loss Rate: {self.lost_stats[-1]:.2f}")
            
            weak_opp_env = env.update_opponent_model(h_env.BasicOpponent(weak=True))

            avg_rew_stats_weak_opp, avg_touch_stats_weak_opp, avg_won_stats_weak_opp, avg_lost_stats_weak_opp = custom_evaluate_policy(
                weak_opp_env,
                self.model,
                n_eval_episodes=self.n_eval_episodes
            )
            
            self.rew_stats_weak_opp.append(avg_rew_stats_weak_opp)
            self.touch_stats_weak_opp.append(avg_touch_stats_weak_opp)
            self.won_stats_weak_opp.append(avg_won_stats_weak_opp)
            self.lost_stats_weak_opp.append(avg_lost_stats_weak_opp)

            print("weak opponent:")
            print(f"Avg Reward: {self.rew_stats_weak_opp[-1]:.2f}, Touch Rate: {self.touch_stats_weak_opp[-1]:.2f}, "
                f"Win Rate: {self.won_stats_weak_opp[-1]:.2f}, Loss Rate: {self.lost_stats_weak_opp[-1]:.2f}")


            if self.custom_log_path is not None:
                assert isinstance(self.rew_stats, list)
                assert isinstance(self.touch_stats, list)
                assert isinstance(self.won_stats, list)
                assert isinstance(self.lost_stats, list)
                np.savez(
                    self.custom_log_path,
                    timesteps=self.evaluations_timesteps,
                    rew_stats=self.rew_stats,
                    touch_stats=self.touch_stats,
                    won_stats=self.won_stats,
                    lost_stats=self.lost_stats,
                    won_stats_weak_opp = self.won_stats_weak_opp,
                    lost_stats_weak_opp = self.lost_stats_weak_opp,
                    touch_stats_weak_opp = self.touch_stats_weak_opp,
                    rew_stats_weak_opp = self.rew_stats_weak_opp
                )

        return result
    
# from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

class ChangeModelParameterCallback(BaseCallback):
    def __init__(self, lr=None, buffer_size=None, verbose = 0):
        super().__init__(verbose)
        self.lr = lr
        self.buffer_size = lr


    def init_callback(self, model):
        super().init_callback(model)
        # model is set now and we can adjust parameters
        if self.lr is not None:
            print("Adjusted learning rate to ", self.lr)
            self.model.learning_rate = self.lr
        if self.buffer_size is not None:
            # create new empty buffer
            print("Created new replay buffer with size ", self.buffer_size)
            self.model.buffer_size = self.buffer_size
            self.model.replay_buffer = self.model.replay_buffer_class(
                self.model.buffer_size,
                self.model.observation_space,
                self.model.action_space,
                device=self.model.device,
                n_envs=self.model.n_envs,
                optimize_memory_usage=self.model.optimize_memory_usage,
                # **replay_buffer_kwargs,
            )


# TODO adjust this callback to use for dividing learning rate
# class StopTrainingOnNoModelImprovement(BaseCallback):
#     """
#     Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.

#     It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

#     It must be used with the ``EvalCallback``.

#     :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
#     :param min_evals: Number of evaluations before start to count evaluations without improvements.
#     :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
#     """

#     parent: EvalCallback

#     def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
#         super().__init__(verbose=verbose)
#         self.max_no_improvement_evals = max_no_improvement_evals
#         self.min_evals = min_evals
#         self.last_best_mean_reward = -np.inf
#         self.no_improvement_evals = 0

#     def _on_step(self) -> bool:
#         assert self.parent is not None, "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

#         continue_training = True

#         if self.n_calls > self.min_evals:
#             if self.parent.best_mean_reward > self.last_best_mean_reward:
#                 self.no_improvement_evals = 0
#             else:
#                 self.no_improvement_evals += 1
#                 if self.no_improvement_evals > self.max_no_improvement_evals:
#                     continue_training = False

#         self.last_best_mean_reward = self.parent.best_mean_reward

#         if self.verbose >= 1 and not continue_training:
#             print(
#                 f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
#             )

#         return continue_training



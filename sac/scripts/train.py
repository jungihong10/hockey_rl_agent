from os.path import join
from os import makedirs

import gymnasium as gym
import hockey.hockey_env as h_env

import numpy as np
from torch.nn import ELU, ReLU
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from src.preprocess import SelfPlayWithGivenOpponent, PreprocessedStateSpace
from src.utils import load_config, CustomCheckpointCallback, ChangeModelParameterCallback

def train(config_path, dest_path, model_path=None):
    print("loading config from ", config_path)
    config = load_config(config_path)

    if config['HER']:
        if config['HER_version'] is not None:
            if config['HER_version'] == 1:
                print(f"Import HER V0")
                from src.her import HockeyHerEnv
            elif config['HER_version'] == 3:
                print(f"Import HER V3")
                from src.her3 import HockeyHerEnv
            else:
                raise ValueError

    if config['reshaped_rewards']:
        if config['reshaped_rewards_version'] is not None:
            if config['reshaped_rewards_version'] == 0:
                print(f"Import reshaped_rewards V0")
                from src.preprocess import RewardShapingClass0 as RewardShapingClass
            elif config['reshaped_rewards_version'] == 1:
                print(f"Import reshaped_rewards V1")
                from src.preprocess import RewardShapingClass1 as RewardShapingClass
            elif config['reshaped_rewards_version'] == 2:
                print(f"Import reshaped_rewards V2")
                from src.preprocess import RewardShapingClass2 as RewardShapingClass
            elif config['reshaped_rewards_version'] == 3:
                print(f"Import reshaped_rewards V3")
                from src.preprocess import RewardShapingClass3 as RewardShapingClass
            else:
                raise ValueError

    env = h_env.HockeyEnv(verbose=False)

    # if config['monitor']:
    #     env = Monitor(env, filename=dest_path, allow_early_resets=True) # TODO this is not working here, csv will be empty!
    if config['enriched_obs']:
        print('use enriched observations.')
        env = PreprocessedStateSpace(env) # TODO make compatible with HockeyHerEnv, accesses states
    if config['reshaped_rewards']:
        print('use reshaped rewards')
        env = RewardShapingClass(env)
    if config['HER']:
        print('use HER')
        env = HockeyHerEnv(env)
    if config['monitor']: # TODO print?
        env = Monitor(env, filename=dest_path, allow_early_resets=True)
    
    train_player2 = h_env.BasicOpponent(weak=config["weak_opponent"])
    train_env = SelfPlayWithGivenOpponent(env, player_two_model=train_player2)

    eval_player2 = h_env.BasicOpponent(weak=config["weak_opponent"])
    eval_env = SelfPlayWithGivenOpponent(env, player_two_model=eval_player2)
    
    # CALLBACKS
    # Use deterministic actions for evaluation
    eval_callback = CustomCheckpointCallback(eval_env, best_model_save_path=dest_path,
                             log_path=dest_path, eval_freq=eval(config["eval_freq"]), n_eval_episodes=config["n_eval_episodes"],
                             deterministic=True, render=False)
    # Save a checkpoint every 1000 steps
    makedirs(join(dest_path, "checkpoints"), exist_ok=True)
    checkpoint_callback = CheckpointCallback(
            save_freq=eval(config["checkpoint_freq"]),
            save_path=join(dest_path, "checkpoints"),
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    

    if config['HER']:
        sac_dict = dict(
            policy = "MultiInputPolicy",
            replay_buffer_class = HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=config["n_sampled_goal"],
                goal_selection_strategy=config["goal_selection_strategy"]
            )
        )
    else:
        sac_dict = dict(
            policy = "MlpPolicy"
        )

    train_freq = eval(config["train_freq"])
    if not isinstance(train_freq, tuple):
        train_freq = int(train_freq)
    # use hyperaparameters from config:
    if model_path is None:
        model = SAC(
            env = train_env,
            verbose= config["verbose"],
            buffer_size=int(eval(config["buffer_size"])),
            learning_rate=eval(config["learning_rate"]),
            gamma=config["gamma"],
            batch_size=config["batch_size"],
            policy_kwargs=dict(
                net_arch=config["net_arch"],
                activation_fn = eval(config["activation_fn"])),
            learning_starts = config["learning_starts"],
            train_freq= train_freq,
            gradient_steps=config["gradient_steps"],
            action_noise=eval(config['action_noise']),
            **sac_dict
        )
    else:
        model = SAC.load(model_path, env=train_env, print_system_info=True)
        print("Model loaded from existing path: ", model_path)
        callbacks.callbacks.append(
            ChangeModelParameterCallback(lr=config["learning_rate"], buffer_size=config["buffer_size"])
        )


    print("start learning")
    model.learn(
        total_timesteps=eval(config["total_timesteps"]),
        log_interval=config["log_interval"],
        callback=callbacks)
     

    # TODO what about winning rate, currently wie have rewards -1/0/1 so we cannot really get true winning rate



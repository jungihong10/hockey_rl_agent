from os.path import join

import gymnasium as gym
import hockey.hockey_env as h_env

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from torch.nn import ELU, ReLU
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.preprocess import SelfPlayWithGivenOpponent, PreprocessedStateSpace
from src.her import HockeyHerEnv
from src.utils import load_config, custom_evaluate_policy
from src.plot import plot_heatmap


def eval(config_path, dest_path, model_name='best_model', only_gif=False, additional_bot_evaluation=False, make_heatmap=True):
     # after learning, make gif of best_model performance

    config = load_config(config_path)

    if 'HER'in config and config['HER']:
        if config['HER_version'] is not None:
            if config['HER_version'] == 1:
                print(f"Import HER V0")
                from src.her import HockeyHerEnv
            elif config['HER_version'] == 3:
                print(f"Import HER V3")
                from src.her3 import HockeyHerEnv
            else:
                raise ValueError

    if 'reshaped_rewards' in config and config['reshaped_rewards']:
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
    
    if 'enriched_obs' in config and config['enriched_obs']:
        print('use enriched observations.')
        env = PreprocessedStateSpace(env) # TODO make compatible with HockeyHerEnv, accesses states
    if 'reshaped_rewards' in config and config['reshaped_rewards']:
        print('use reshaped rewards')
        env = RewardShapingClass(env)
    if 'HER' in config and config['HER']:
        print('use HER')
        env = HockeyHerEnv(env)
    # if config['monitor']: # -> IO operation on closed file error
        # env = Monitor(env, filename=dest_path, allow_early_resets=True)
    
    eval_player2 = h_env.BasicOpponent(weak=config["weak_opponent"])
    eval_env = SelfPlayWithGivenOpponent(env, player_two_model=eval_player2)


    env = eval_env
    unwrapped_env = env.unwrapped

    best_model_path = join(dest_path, model_name)
    # best_model_path = join(dest_path, "../../model/SAC")
    model = SAC.load(best_model_path, env=env)
    
    if config["make_gif"]:
        from PIL import Image
        
        images = []
        obs, info = env.reset()
        img = unwrapped_env.render(mode="rgb_array")

        for i in range(500):
            images.append(Image.fromarray(img))
            a1, _states = model.predict(obs, deterministic=True)   # What does deterministic=True do? same seed?
            obs, r, d, _, info = env.step(a1)
            img = unwrapped_env.render(mode="rgb_array")
            if d: 
                env.reset()
        
        env.close()

        # Save as GIF
        images[0].save(
            best_model_path + ".gif",
            save_all=True,
            append_images=images[1::2], 
            duration=20,  # Adjust speed (1000 ms / fps) # below 20 is not supported:( -> only take every 2nd image
            loop=0  # Infinite loop
        )

    if not only_gif:
        if config["plot_win_rate"]:
            make_plot(dest_path=dest_path, model_name=model_name)

    if additional_bot_evaluation:
        ## evaluate episodes of model
        n_eval_episodes = 100
        env.update_opponent_model(h_env.BasicOpponent(weak=True))
        _, _, win_rate, lost_rate = custom_evaluate_policy(env, model, n_eval_episodes)
        print(f"In {n_eval_episodes} the agent had a winnning rate of {win_rate} and a loosing rate of {lost_rate}")

        env.update_opponent_model(h_env.BasicOpponent(weak=False))
        _, _, win_rate, lost_rate = custom_evaluate_policy(env, model, n_eval_episodes)
        print(f"In {n_eval_episodes} the agent had a winnning rate of {win_rate} and a loosing rate of {lost_rate}")

    if make_heatmap:
        env.update_opponent_model(h_env.BasicOpponent(weak=False))
        fig, axes = plot_heatmap(
            make_heatmap_data(env=env, model=model, n_eval_episodes=100, deterministic_agent=True)
        )
        plt.savefig(join(dest_path, f"heatmap_{model_name}_deterministic_agent.png"))
        plt.close()
        fig, axes = plot_heatmap(
            make_heatmap_data(env=env, model=model, n_eval_episodes=100, deterministic_agent=False)
        )
        plt.savefig(join(dest_path, f"heatmap_{model_name}_non_deterministic_agent.png"))
        plt.close()
        env.update_opponent_model(h_env.BasicOpponent(weak=True))
        fig, axes = plot_heatmap(
            make_heatmap_data(env=env, model=model, n_eval_episodes=100, deterministic_agent=True)
        )
        plt.savefig(join(dest_path, f"heatmap_{model_name}_deterministic_agent_weak_opp.png"))
        plt.close()
        


def make_plot(dest_path, model_name, rolling_mean_window=10, max_timesteps=1e6):

    eval_dict = np.load(join(dest_path, "evaluations.npz")) # saved by EvalCallback
    timesteps2 = eval_dict["timesteps"]
    results = eval_dict["results"]
    ep_lengths = eval_dict["ep_lengths"]

    avg_ep_lengths = [np.mean(eval_length) for eval_length in ep_lengths]

    # custom
    eval_dict = np.load(join(dest_path, "custom_evaluations.npz")) # saved by EvalCallback
    timesteps = eval_dict["timesteps"]
    avg_reward = eval_dict["rew_stats"]
    avg_touch_per_episode = eval_dict["touch_stats"]
    # print(avg_touch_per_episode)
    win_rates = eval_dict["won_stats"]
    lose_rates = eval_dict["lost_stats"]
    draw_rates = 1 -  win_rates - lose_rates

    reward_rolling_mean = pd.Series(avg_reward).rolling(window=rolling_mean_window, min_periods=1).mean() 
    win_rate_rolling_mean = pd.Series(win_rates).rolling(window=rolling_mean_window, min_periods=1).mean() 
    draw_rate_rolling_mean = pd.Series(draw_rates).rolling(window=rolling_mean_window, min_periods=1).mean() 
    lose_rate_rolling_mean = pd.Series(lose_rates).rolling(window=rolling_mean_window, min_periods=1).mean() 
    touch_per_episode_rolling_mean = pd.Series(avg_touch_per_episode).rolling(window=rolling_mean_window, min_periods=1).mean() 
    ep_length_rolling_mean = pd.Series(avg_ep_lengths).rolling(window=rolling_mean_window, min_periods=1).mean() 
    


    # SINGLE REWARD ################################
    # Plot reward
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    # original data
    plt.plot(timesteps, avg_reward, color='black', alpha=0.2, label='Reward')
    # rolling mean
    plt.plot(timesteps, reward_rolling_mean, color='darkgreen', linewidth=2, label=f'{rolling_mean_window}-point Avg')
    # plt.plot(timesteps, win_rates, marker="o", linestyle="-", label="Win Rate")
    # plt.plot(timesteps, rew_stats, marker="o", linestyle="-", label="Reward")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Reward Evolution")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(dest_path, f"reward_single_plot_{model_name}.png"))
    plt.close()


    # REWARD AND WIN/LOSE/DRAW RATE ################################################
    # Plot touch
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns    plt.subplot(1, 1, 1)

    # Left: Reward over time
    axes[0].plot(timesteps, avg_reward, color='black', alpha=0.2, label='Reward')
    axes[0].plot(timesteps, reward_rolling_mean, color='darkgreen', linewidth=2, label=f'{rolling_mean_window}-point Avg')
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Reward Statistics Over Time')
    axes[0].legend()
    # histogram
    bar_width = timesteps[1]-timesteps[0]
    axes[1].bar(timesteps, win_rate_rolling_mean, color='green', label='Win Rate', width=bar_width)
    axes[1].bar(timesteps, draw_rate_rolling_mean, bottom=win_rate_rolling_mean, color='blue', label='Draw Rate', width=bar_width)
    axes[1].bar(timesteps, lose_rate_rolling_mean, bottom=win_rate_rolling_mean + draw_rate_rolling_mean, color='red', label='Lose Rate', width=bar_width)

    # Labels and legend
    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Rate')
    axes[1].set_title('Win, Lose, and Draw Rate per Timestep')
    axes[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(join(dest_path, f"reward_and_performance_histogram_plot_{model_name}.png"))
    plt.close()

    # Extensive stats with episode length and touch rate ################################################
    # Plot touch
    fig, axes = plt.subplots(2, 2, figsize=(12, 5))  # 1 row, 2 columns    plt.subplot(1, 1, 1)
    axes[0][0].set_xlim(0, max_timesteps)
    axes[0][1].set_xlim(0, max_timesteps)
    axes[1][0].set_xlim(0, max_timesteps)
    axes[1][1].set_xlim(0,  max_timesteps)
    axes[0][0].set_ylim(-20, 10) # reward
    axes[0][1].set_ylim(0, 1) # histo
    axes[1][0].set_ylim(0, 2.5) # touches
    axes[1][1].set_ylim(0,  250) # max timesteps
    
    # Left: Reward over time
    axes[0][0].plot(timesteps, avg_reward, color='black', alpha=0.2, label='Reward')
    axes[0][0].plot(timesteps, reward_rolling_mean, color='darkgreen', linewidth=2, label=f'{rolling_mean_window}-point Avg')
    axes[0][0].set_xlabel('Timesteps')
    axes[0][0].set_ylabel('Reward')
    axes[0][0].set_title('Reward Statistics Over Time')
    axes[0][0].legend()
    # histogram
    bar_width = timesteps[1]-timesteps[0]
    axes[0][1].bar(timesteps, win_rate_rolling_mean, color='green', label='Win Rate', width=bar_width)
    axes[0][1].bar(timesteps, draw_rate_rolling_mean, bottom=win_rate_rolling_mean, color='blue', label='Draw Rate', width=bar_width)
    axes[0][1].bar(timesteps, lose_rate_rolling_mean, bottom=win_rate_rolling_mean + draw_rate_rolling_mean, color='red', label='Lose Rate', width=bar_width)
    axes[0][1].set_xlabel('Timesteps')
    axes[0][1].set_ylabel('Rate')
    axes[0][1].set_title('Win, Lose, and Draw Rate per Timestep')
    axes[0][1].legend()

    axes[1][0].plot(timesteps, avg_touch_per_episode, color='black', alpha=0.2, label='Puck Touches')
    axes[1][0].plot(timesteps, touch_per_episode_rolling_mean, color='brown', linewidth=2, label=f'{rolling_mean_window}-point Avg')
    axes[1][0].set_xlabel('Timesteps')
    axes[1][0].set_ylabel('Puck Touches')
    axes[1][0].set_title('Avg Puck Touches per Episode')
    axes[1][0].legend()

    axes[1][1].plot(timesteps2, avg_ep_lengths, color='black', alpha=0.2, label='Episode length')
    axes[1][1].plot(timesteps2, ep_length_rolling_mean, color='yellow', linewidth=2, label=f'{rolling_mean_window}-point Avg')
    axes[1][1].set_xlabel('Timesteps')
    axes[1][1].set_ylabel('Episode length')
    axes[1][1].set_title('Avg Episode Length')
    axes[1][1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(join(dest_path, f"extensive_stats_{model_name}.png"))
    plt.close()

    ## PLOT WEAK agent, STRONG agent performance ###################################
    if 'won_stats_weak_opp' in eval_dict:
        # weak agent stats available, hence plot

        win_rate_weak_opp = eval_dict['won_stats_weak_opp']
        lose_rate_weak_agent = eval_dict['lost_stats_weak_opp']
        draw_rate_weak_agent = 1 - win_rate_weak_opp - lose_rate_weak_agent

        win_rate_weak_opp_rolling_mean = pd.Series(win_rate_weak_opp).rolling(window=rolling_mean_window, min_periods=1).mean() 
        lose_rate_weak_opp_rolling_mean = pd.Series(lose_rate_weak_agent).rolling(window=rolling_mean_window, min_periods=1).mean() 
        draw_rate_weak_opp_rolling_mean = pd.Series(draw_rate_weak_agent).rolling(window=rolling_mean_window, min_periods=1).mean() 

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns    plt.subplot(1, 1, 1)

        axes[0].plot(timesteps, win_rate_rolling_mean, color='darkgreen', alpha=1, label='strong opponent') # only plot rolling mean
        axes[0].plot(timesteps, win_rate_weak_opp_rolling_mean, color='green', alpha=1, label='weak opponent') # only plot rolling mean
        # axes[0].plot(timesteps, reward_rolling_mean, color='darkgreen', linewidth=2, label=f'{rolling_mean_window}-point Avg')
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel(f'{rolling_mean_window}-point average win rate')
        axes[0].set_title('Win Rate')
        axes[0].legend()
       
        axes[1].plot(timesteps, lose_rate_rolling_mean, color='darkgreen', alpha=1, label='strong opponent') # only plot rolling mean
        axes[1].plot(timesteps, lose_rate_weak_opp_rolling_mean, color='green', alpha=1, label='weak opponent') # only plot rolling mean
        # axes[1].plot(timesteps, reward_rolling_mean, color='darkgreen', linewidth=2, label=f'{rolling_mean_window}-point Avg')
        axes[1].set_xlabel('Timesteps')
        axes[1].set_ylabel(f'{rolling_mean_window}-point average win rate')
        axes[1].set_title('Win Rate')
        axes[1].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig(join(dest_path, f"win_rates_strong_weak_opp{model_name}.png"))
        plt.close()


def make_heatmap_data(env, model, n_eval_episodes=1000, deterministic_agent=True):
    # IDX_PLAYER_1_POS_X = 0
    # IDX_PLAYER_1_POS_Y = 1
    # IDX_PLAYER_2_POS_X = 6
    # IDX_PLAYER_2_POS_Y = 7
    # IDX_PUCK_POS_X = 12
    # IDX_PUCK_POS_Y = 13

    # positions_player_1 = []
    # positions_player_2 = []
    # positions_puck = []
    observations = []
    unwrapped_env = env.unwrapped
    obs, info = env.reset()
    for ep in range(n_eval_episodes):
        for i in range(250):
            a1, _states = model.predict(obs, deterministic=False)   # What happens to heatmap if not deterministic????? TODO
            obs, r, d, _, info = env.step(a1)
            observations.append(obs)
            # positions_player_1.append((obs[IDX_PLAYER_1_POS_X, IDX_PLAYER_1_POS_Y]))
            # positions_player_2.append((obs[IDX_PLAYER_2_POS_X, IDX_PLAYER_2_POS_Y]))
            # positions_puck.append((obs[IDX_PUCK_POS_X], obs[IDX_PUCK_POS_Y]))

            if d: 
                env.reset()
    env.close()

    return observations



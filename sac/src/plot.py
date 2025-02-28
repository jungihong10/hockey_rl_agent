# plotting: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
# stable-baselines3: https://github.com/DLR-RM/stable-baselines3/blob/c5c29a32d961be692e08ff49c94d2485ac40cb8a/stable_baselines3/common/results_plotter.py#L4
# offical doc stable-baseline: https://stable-baselines.readthedocs.io/en/master/misc/results_plotter.html

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy

from os.path import dirname, join
import matplotlib.pyplot as plt
import pandas as pd

# LOG_DIR="../models/"
# full_path = join(dirname(__file__), LOG_DIR)

# # Helper from the library
# results_plotter.plot_results(
#     [full_path], 1e5, results_plotter.X_TIMESTEPS, "Laser-Hockey 2x128"
# )
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

# DISCLAIMER: Visualization adapted from ChatGPT
# Define field dimensions
SCALE = 60.0  
VIEWPORT_W = 600
VIEWPORT_H = 480
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CENTER_X = W / 2
CENTER_Y = H / 2

IDX_PLAYER_1_POS_X = 0
IDX_PLAYER_1_POS_Y = 1
IDX_PLAYER_2_POS_X = 6
IDX_PLAYER_2_POS_Y = 7
IDX_PUCK_POS_X = 12
IDX_PUCK_POS_Y = 13

def plot_heatmap(observations: np.ndarray, bins=20):
    """
    Plots heatmaps showing the positional distribution of player1, player2, and puck.
    
    Args:
        observations (np.ndarray): Sequence of observations (N x state_size).
        bins (int): Number of bins for the heatmap.
    """
    if not isinstance(observations, np.ndarray):
        observations = np.array(observations)
    fig, axes = plt.subplots(1, 3, figsize=(W, H/3))
    titles = ['Player 1', 'Player 2', 'Puck']
    indices = [
        (IDX_PLAYER_1_POS_X, IDX_PLAYER_1_POS_Y),
        (IDX_PLAYER_2_POS_X, IDX_PLAYER_2_POS_Y),
        (IDX_PUCK_POS_X, IDX_PUCK_POS_Y)]
    
    for i, (x_idx, y_idx) in enumerate(indices):
        heatmap, xedges, yedges = np.histogram2d(
            observations[:, x_idx].flatten(), observations[:, y_idx].flatten(), bins=bins, range=[[-W/2, W/2], [-H/2, H/2]]
        )
        # axes[i].set_xlim(-W/2, W/2) # we can also use this instead of extent, but than from -W/2-W/2
        # axes[i].set_ylim(-H/2, H/2)
        axes[i].imshow(heatmap.T, origin='lower', cmap='hot', extent=[0, W, 0, H])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
    
    plt.tight_layout()
    return fig, axes

def plot_trajectories(observations: np.ndarray):
    """
    Plots the trajectories of player1, player2, and the puck.
    
    Args:
        observations (np.ndarray): Array of sequences of observations (N_sequences, T, state_size).
    """
    colors = ['red', 'blue', 'black']
    labels = ['Player 1', 'Player 2', 'Puck']
    
    plt.figure(figsize=(8, 8))
    for i, (x_idx, y_idx) in enumerate([(0, 1), (2, 3), (4, 5)]):
        for sequence in observations:
            plt.plot(sequence[:, x_idx], sequence[:, y_idx], color=colors[i], alpha=0.5)
    
    plt.xlim([0, W])
    plt.ylim([0, H])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectories of Players and Puck')
    plt.legend(labels)
    plt.grid()
    plt.show()

def plot_reward_and_winning_rate(fig, axes, eval_dict, rolling_mean_window=10, label="missing label"):

    # custom
    timesteps = eval_dict["timesteps"]
    avg_reward = eval_dict["rew_stats"]
    win_rates = eval_dict["won_stats"]

    reward_rolling_mean = pd.Series(avg_reward).rolling(window=rolling_mean_window, min_periods=1).mean() 
    win_rate_rolling_mean = pd.Series(win_rates).rolling(window=rolling_mean_window, min_periods=1).mean() 


    axes[0].plot(timesteps, reward_rolling_mean, alpha=1, label=label) # only plot rolling mean
    axes[1].plot(timesteps, win_rate_rolling_mean, alpha=1, label=label) # only plot rolling mean

    return fig, axes

def plot_reward(ax, eval_dict, rolling_mean_window=10, label="missing label"):

    # custom
    timesteps = eval_dict["timesteps"]
    avg_reward = eval_dict["rew_stats"]

    reward_rolling_mean = pd.Series(avg_reward).rolling(window=rolling_mean_window, min_periods=1).mean() 


    ax.plot(timesteps, reward_rolling_mean, alpha=1, label=label) # only plot rolling mean

    return ax
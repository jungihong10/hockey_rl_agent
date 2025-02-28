# plotting: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
# stable-baselines3: https://github.com/DLR-RM/stable-baselines3/blob/c5c29a32d961be692e08ff49c94d2485ac40cb8a/stable_baselines3/common/results_plotter.py#L4
# offical doc stable-baseline: https://stable-baselines.readthedocs.io/en/master/misc/results_plotter.html

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy

from os.path import dirname, join
import matplotlib.pyplot as plt

LOG_DIR="../models/"
full_path = join(dirname(__file__), LOG_DIR)

# Helper from the library
results_plotter.plot_results(
    [full_path], 1e5, results_plotter.X_TIMESTEPS, "Laser-Hockey 2x128"
)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# DISCLAIMER: Visualization adapted from ChatGPT
# Define field dimensions
SCALE = 60.0  
VIEWPORT_W = 600
VIEWPORT_H = 480
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

def plot_heatmaps(observations: np.ndarray, bins=20):
    """
    Plots heatmaps showing the positional distribution of player1, player2, and puck.
    
    Args:
        observations (np.ndarray): Sequence of observations (N x state_size).
        bins (int): Number of bins for the heatmap.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Player 1', 'Player 2', 'Puck']
    indices = [(0, 1), (2, 3), (4, 5)]
    
    for i, (x_idx, y_idx) in enumerate(indices):
        heatmap, xedges, yedges = np.histogram2d(
            observations[:, x_idx].flatten(), observations[:, y_idx].flatten(), bins=bins, range=[[0, W], [0, H]]
        )
        axes[i].imshow(heatmap.T, origin='lower', cmap='hot', extent=[0, W, 0, H])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
    
    plt.tight_layout()
    plt.show()

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

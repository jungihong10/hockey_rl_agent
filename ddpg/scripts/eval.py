import time
import sys
import os
import numpy as np
import gymnasium as gym
import hockey.hockey_env as h_env
from stable_baselines3 import DDPG
from argparse import ArgumentParser

sys.path.insert(0, '.')
sys.path.insert(1, '..')
#from utils.utils import *
from base.evaluator import evaluate


parser = ArgumentParser()
parser.add_argument("--mode", type=str, choices=["normal", "shooting", "defense"], default="normal")
opts = parser.parse_args()


def evaluate(model, env, opponent, eval_episodes=10, render=False):
    rew_stats = []
    touch_stats = []
    won_stats = []
    lost_stats = []

    for episode_counter in range(eval_episodes):
        total_reward = 0
        ob, _ = env.reset()
        obs_agent2 = env.obs_agent_two()

        touch = 0
        won = 0
        lost = 0

        for step in range(env.max_timesteps):
            if render:
                time.sleep(0.01)
                env.render()

            a1, _ = model.predict(ob, deterministic=True)  # Predict action for agent
            a2 = opponent.act(obs_agent2)  # Opponent action

            ob, reward, done, _, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()

            total_reward += reward
            touch += int(info['reward_touch_puck'] > 0)

            if done:
                won = int(env.winner == 1)
                lost = int(env.winner == -1)
                break

        rew_stats.append(total_reward)
        touch_stats.append(touch)
        won_stats.append(won)
        lost_stats.append(lost)

    env.close()

    print(f"Avg Reward: {np.mean(rew_stats):.2f}, Touch Rate: {np.mean(touch_stats):.2f}, "
          f"Win Rate: {np.mean(won_stats):.2f}, Loss Rate: {np.mean(lost_stats):.2f}")

    return np.mean(rew_stats), np.mean(touch_stats), np.mean(won_stats), np.mean(lost_stats)


if __name__ == "__main__":

    env = h_env.HockeyEnv()
    opponent = h_env.BasicOpponent(weak=False)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models15/oldies/ddpg_hockey_1500000"))

    print(f"Loading model from {model_path}")
    model = DDPG.load(model_path)

    evaluate(model, env, opponent, eval_episodes=4, render=True)

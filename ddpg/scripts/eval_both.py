import time
import sys
import os
import json
import numpy as np
import gymnasium as gym
import hockey.hockey_env as h_env
from stable_baselines3 import DDPG

sys.path.insert(0, '.')
sys.path.insert(1, '..')

# Evaluation Function
def evaluate(model, env, opponent, eval_episodes=50, render=False):
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

    return {
        "Avg Reward": float(np.mean(rew_stats)),
        "Touch Rate": float(np.mean(touch_stats)),
        "Win Rate": float(np.mean(won_stats)),
        "Loss Rate": float(np.mean(lost_stats)),
    }

# Main Execution
if __name__ == "__main__":

    env = h_env.HockeyEnv()
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models15/"))
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]

    results = {}

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        print(f"\nLoading model from {model_path}")
        model = DDPG.load(model_path)

        # Evaluate against weak opponent
        print(f"Evaluating {model_file} against Weak Opponent")
        weak_opponent = h_env.BasicOpponent(weak=True)
        weak_results = evaluate(model, env, weak_opponent, eval_episodes=50, render=False)

        # Evaluate against strong opponent
        print(f"Evaluating {model_file} against Strong Opponent")
        strong_opponent = h_env.BasicOpponent(weak=False)
        strong_results = evaluate(model, env, strong_opponent, eval_episodes=50, render=False)

        # Store results
        results[model_file] = {
            "Weak Opponent": weak_results,
            "Strong Opponent": strong_results
        }

    # Save results to JSON file
    results_file = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {results_file}")

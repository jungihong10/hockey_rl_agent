import gymnasium as gym
import hockey.hockey_env as h_env
from ddpg import DDPG
from preprocess import SelfPlayWithGivenOpponent
from eval import evaluate  # Import evaluation function
import json  # To save evaluation results

# Initialize base environment
env = h_env.HockeyEnv()
wrapped_env = SelfPlayWithGivenOpponent(env, player_two_model=None)  # Start with basic opponent

# Pretraining (if needed)
policy_kwargs = {"net_arch": [256, 256]}
model = DDPG(
    "MlpPolicy", 
    wrapped_env, 
    policy_kwargs=policy_kwargs, 
    learning_rate=0.0001, 
    verbose=1)
    #batch_size=128,
    #buffer_size=100000,
    #tau=0.005,
    #train_freq=(1, "episode"),

# Dictionary to store results
results = {"timesteps": [], "win_rates": []}

# Self-play loop
total_timesteps = 2_000_000  # Total training steps
eval_interval = 100_000  # Evaluate every X timesteps

eval_episodes = 50  # Number of episodes for evaluation

wrapped_env.update_opponent_model(h_env.BasicOpponent(weak=True))  # Ensure an opponent is set

# Train agent and evaluate at intervals
for t in range(0, total_timesteps, eval_interval):
    model.learn(total_timesteps=eval_interval, log_interval=4)

    # Evaluate performance
    opponent = h_env.BasicOpponent(weak=True)
    _, _, win_rate, _ = evaluate(model, env, opponent, eval_episodes)

    # Store results
    results["timesteps"].append(t + eval_interval)
    results["win_rates"].append(win_rate)

    print(f"Trained {t + eval_interval} timesteps. Win Rate: {win_rate:.2f}")

    # Save model after every eval_interval
    model_path = f"models/ddpg_hockey_{t + eval_interval}"
    model.save(model_path)
    print(f"Model saved at {model_path}")

# Save final training results
with open("training_winrate_log.json", "w") as f:
    json.dump(results, f)

# Cleanup
del model
env.close()

# Reload the last trained model for evaluation
model_path = f"models/ddpg_hockey_{total_timesteps}"
model = DDPG.load(model_path)

# Final evaluation
opponent = h_env.BasicOpponent(weak=True)
mean_reward, touch_rate, win_rate, loss_rate = evaluate(model, env, opponent, eval_episodes=50)

print(f"Final Evaluation: Mean Reward: {mean_reward}, Touch Rate: {touch_rate}, Win Rate: {win_rate}, Loss Rate: {loss_rate}")

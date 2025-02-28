import gymnasium as gym
import hockey.hockey_env as h_env
from ddpg import DDPG
from preprocess import SelfPlayWithGivenOpponent
from eval import evaluate
import json
import shutil  # To copy saved models

class DDPGOpponent:
    def __init__(self, model_path):
        self.model = DDPG.load(model_path)

    def act(self, obs):
        result = self.model.predict(obs, deterministic=True)
        action = result[0]  # Take the first element as the action
        return action  # act() returns just the action

    def predict(self, obs, deterministic=True):
        result = self.model.predict(obs, deterministic)
        action = result[0]  # Always take the first element as the action
        return action, None  # Return exactly two values (action, dummy)

# Load or initialize model
saved_model_path = "models/ddpg_hockey_selfplay_start.zip"  # Path to a pre-trained model (optional)
try:
    model = DDPG.load(saved_model_path)
    print(f"Loaded model from {saved_model_path}")
except:
    print("No pre-trained model found. Training from scratch.")
    policy_kwargs = {"net_arch": [256, 256]}
    env = h_env.HockeyEnv()
    model = DDPG(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,
        verbose=1,
    )

# Initialize base environment
env = h_env.HockeyEnv()
wrapped_env = SelfPlayWithGivenOpponent(env, player_two_model=None)  # Placeholder for self-play

model.set_env(wrapped_env)

# Dictionary to store results
results = {"timesteps": [], "win_rates": []}

# Training parameters
total_timesteps = 2_000_000  # Total training steps
eval_interval = 100_000  # Evaluate every X timesteps
opponent_update_interval = 300_000  # Update opponent every Y timesteps
eval_episodes = 50  # Number of episodes for evaluation

# Self-play initialization: Set opponent as an old version of the model
opponent_path = "models/ddpg_hockey_opponent.zip"
model.save(opponent_path)  # Save initial model as opponent
wrapped_env.update_opponent_model(DDPGOpponent(opponent_path))


# Train agent and evaluate at intervals
for t in range(0, total_timesteps, eval_interval):
    model.learn(total_timesteps=eval_interval, log_interval=4)

    # Evaluate performance against the current opponent (past version of itself)
    opponent = DDPGOpponent(opponent_path)
    _, _, win_rate, _ = evaluate(model, env, opponent, eval_episodes)

    # Store results
    results["timesteps"].append(t + eval_interval)
    results["win_rates"].append(win_rate)

    print(f"Trained {t + eval_interval} timesteps. Win Rate vs Past Self: {win_rate:.2f}")

    # Save model after every eval_interval
    model_path = f"models/ddpg_hockey_{t + eval_interval}.zip"
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Every opponent_update_interval, update the self-play opponent
    if (t + eval_interval) % opponent_update_interval == 0:
        shutil.copy(model_path, opponent_path)  # Replace opponent model
        wrapped_env.update_opponent_model(DDPGOpponent(opponent_path))

        print(f"Updated opponent model to latest at {t + eval_interval} timesteps")

# Save final training results
with open("training_winrate_log.json", "w") as f:
    json.dump(results, f)

# Cleanup
del model
env.close()

# Reload the last trained model for final evaluation
model_path = f"models/ddpg_hockey_{total_timesteps}.zip"
model = DDPG.load(model_path)

# Final evaluation against its latest past self
mean_reward, touch_rate, win_rate, loss_rate = evaluate(model, env, DDPG.load(opponent_path), eval_episodes=50)

print(f"Final Evaluation: Mean Reward: {mean_reward}, Touch Rate: {touch_rate}, Win Rate: {win_rate}, Loss Rate: {loss_rate}")

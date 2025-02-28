import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, 
                 alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        self.alpha = alpha                # Controls how much prioritization is used
        self.beta = beta                  # Importance sampling weight
        self.beta_increment = beta_increment  # Increase beta over time
        self.epsilon = epsilon            # Small constant to prevent zero probability
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.max_priority = 1.0           # Start with max priority

    def add(self, *args, **kwargs):
        index = self.pos  # Position to store the new transition
        super().add(*args, **kwargs)
        self.priorities[index] = self.max_priority  # Assign max priority to new transition

    def sample(self, batch_size, env=None, **kwargs):
        # Compute sampling probabilities
        if self.full:
            probs = self.priorities ** self.alpha
            probs /= probs.sum()
        else:
            probs = self.priorities[:self.pos] ** self.alpha
            probs /= probs.sum()

        # Sample indices based on the computed probabilities
        indices = np.random.choice(len(probs), batch_size, p=probs)
        # Compute importance-sampling weights (not used in training loop yet)
        weights = (len(probs) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize the weights
        self.beta = min(1.0, self.beta + self.beta_increment)  # Increase beta over time

        # Remove any duplicate 'env' from kwargs to avoid conflicts
        kwargs.pop('env', None)
        # Call _get_samples with indices and env
        batch = super()._get_samples(indices, env=env, **kwargs)
        # Return only the batch so that training loop gets a ReplayBufferSamples instance
        return batch

    def update_priorities(self, indices, td_errors):
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, new_priorities.max())

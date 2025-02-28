# this file is to preprocess the environment, e.g. get better state representations from rgb images
# ALREADY PREPROCESSED, not rgb space!!
'''
state representation 1:
- player1: pos, angle, vel  (possibility to split this in v_x, v_y, but probably bad for convergence)
- player2: pos, angle, vel
- puck: pos, vel_x, vel_y
'''

'''
- player1: pos, angle, vel
- player2: pos_wrt_player1, angle_wrt_player1, vel_wrt_player1
- puck: pos_wrt_player1, vel_wrt_player1
'''

import gymnasium as gym
import hockey.hockey_env as h_env
import numpy as np

class EgoActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def action(self, action):
        # opponent should not move.
        return np.hstack([action, np.zeros(4)])


class SelfPlayWithGivenOpponent(gym.ActionWrapper):
    def __init__(self, env, player_two_model):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        if player_two_model is None:
            self.player_two_model = h_env.BasicOpponent()
        else:
            # TODO check if palyer_two_model is model
            self.player_two_model = player_two_model

    def update_opponent_model(self, player_two_model):
        self.player_two_model = player_two_model

    def action(self, action):
        if isinstance(self.player_two_model, h_env.BasicOpponent):
            # if basic opponent, use hand-crafted act method
            action_player_two = self.player_two_model.act(self.env.obs_agent_two())
        else:
            # if model, use predict method
            action_player_two, _ = self.player_two_model.predict(self.env.obs_agent_two(), deterministic=False)  
        return np.hstack([action, action_player_two])



class PreprocessedStateSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def observation(self, obs):
        # Extract raw state information
        px, py, angle, vx, vy, av = obs[:6]  # Player 1
        px2, py2, angle2, vx2, vy2, av2 = obs[6:12]  # Player 2
        puck_x, puck_y, puck_vx, puck_vy = obs[12:16]  # Puck

        # Compute relative positions
        rel_puck_x = puck_x - px
        rel_puck_y = puck_y - py
        rel_puck_vx = puck_vx - vx
        rel_puck_vy = puck_vy - vy

        rel_opponent_x = px2 - px
        rel_opponent_y = py2 - py

        return np.array([
            px, py, angle, vx, vy, av,  # Agent’s state
            rel_puck_x, rel_puck_y, rel_puck_vx, rel_puck_vy,  # Puck relative position
            rel_opponent_x, rel_opponent_y  # Opponent relative position
        ], dtype=np.float32)

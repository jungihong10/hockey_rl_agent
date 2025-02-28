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
from src.her3 import HockeyHerEnv as HockHerEnv3 # TODO version adaptive
from src.her import HockeyHerEnv as HockHerEnv1
import numpy as np
from stable_baselines3.common.monitor import Monitor


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
        self.hockey_env = env
        if not isinstance(env, h_env.HockeyEnv) and not isinstance(env, HockHerEnv3) and not isinstance(env, HockHerEnv1):
            self.hockey_env = env.unwrapped
        # if isinstance(env, Monitor):    # if additionally wrapped we need to unpack environments to access hockey_env
        #     self.hockey_env = self.env.env
        # else:
        #     self.hockey_env = env
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        if player_two_model is None:
            self.player_two_model = h_env.BasicOpponent()
        else:
            # TODO check if palyer_two_model is model
            self.player_two_model = player_two_model

    def update_opponent_model(self, player_two_model):
        self.player_two_model = player_two_model
        return self

    def action(self, action):
        if isinstance(self.player_two_model, h_env.BasicOpponent):
            # if basic opponent, use hand-crafted act method
            action_player_two = self.player_two_model.act(self.hockey_env.obs_agent_two())
        else:
            # if model, use predict method
            action_player_two, _ = self.player_two_model.predict(self.hockey_env.obs_agent_two(), deterministic=False)  
        return np.hstack([action, action_player_two])



class RewardShapingClass2(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.hockey_env = env
        if not isinstance(env, h_env.HockeyEnv) and not isinstance(env, HockHerEnv3) and not isinstance(env, HockHerEnv1):
            self.hockey_env = env.unwrapped

    def reward(self, reward):

        goal_reward = self.hockey_env.winner * 100

        proximity_reward = 0
        info = self.hockey_env._get_info()
        proximity_reward = 4 * info['reward_closeness_to_puck']
        # touch_reward = 5 * info['reward_touch_puck']
            # print(proximity_reward)
        
        r = goal_reward + proximity_reward
        return  r/100


class RewardShapingClass3(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.hockey_env = env
        if not isinstance(env, h_env.HockeyEnv) and not isinstance(env, HockHerEnv3) and not isinstance(env, HockHerEnv1):
            self.hockey_env = env.unwrapped

    def reward(self, reward):

        goal_reward = self.hockey_env.winner * 10

        proximity_reward = 0
        info = self.hockey_env._get_info()
        proximity_reward = 4 * info['reward_closeness_to_puck']
        # touch_reward = 5 * info['reward_touch_puck']
            # print(proximity_reward)
        
        r = goal_reward + proximity_reward
        return  r/10

# ???
class RewardShapingClass1(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.hockey_env = env
        if not isinstance(env, h_env.HockeyEnv) and not isinstance(env, HockHerEnv3) and not isinstance(env, HockHerEnv1):
            self.hockey_env = env.unwrapped

    def reward(self, reward):
        info = self.hockey_env._get_info()

        return  reward + info['reward_closeness_to_puck']


class RewardShapingClass0(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.hockey_env = env
        if not isinstance(env, h_env.HockeyEnv) and not isinstance(env, HockHerEnv3) and not isinstance(env, HockHerEnv1):
            self.hockey_env = env.unwrapped

    def reward(self, reward):
        IDX_PUK_VEL_X = 14
        IDX_PUK_VEL_Y = 15
        IDX_PUK_POS_X = 12
        IDX_PUK_POS_Y = 13
        IDX_P1_POS_X = 0
        IDX_P1_POS_Y = 1


        GOAL_POS_X = (300 + 250) / 60
        GOAL_POS_Y = 480 / 2 / 60

        SCALE = 60.0 
        VIEWPORT_W = 600
        VIEWPORT_H = 480
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        CENTER_X = W / 2
        CENTER_Y = H / 2

        obs = self.hockey_env._get_obs()

        goal_reward = self.hockey_env.winner * 100

        v = np.array([obs[IDX_PUK_VEL_X], obs[IDX_PUK_VEL_Y]])
        rel_pos_goal = np.array([
            GOAL_POS_X -  (obs[IDX_PUK_POS_X] + CENTER_X),
            GOAL_POS_Y - (obs[IDX_PUK_POS_Y] + CENTER_Y)
        ])
        puck_dir_reward = normalized_dot_product(v, rel_pos_goal)
        
        proximity_reward = 0
        if obs[IDX_PUK_POS_X] < 0: # Puck in the half of agent
            info = self.hockey_env._get_info()
            proximity_reward = info['reward_closeness_to_puck']
            # print(proximity_reward)

        return  goal_reward + puck_dir_reward + proximity_reward

import numpy as np

def normalized_dot_product(a, b):
    """Compute the normalized dot product (cosine similarity) between two vectors."""
    a = np.array(a)
    b = np.array(b)
    
    # Compute dot product
    dot_product = np.dot(a, b)
    
    # Compute norms (magnitudes)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0
        # raise ValueError("One of the vectors has zero magnitude, cannot compute normalized dot product.")
    
    # Compute normalized dot product
    return dot_product / (norm_a * norm_b)





class PreprocessedStateSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

    def observation(self, observation):
        """
        Transforms the state vector to include relative positions and velocities.
        
        Args:
            observation (np.ndarray): The original state vector.
            
        Returns:
            np.ndarray: The transformed state vector.
        
        # 0  x pos player one
        # 1  y pos player one
        # 2  angle player one
        # 3  x vel player one
        # 4  y vel player one
        # 5  angular vel player one
        # 6  x player two
        # 7  y player two
        # 8  angle player two
        # 9 y vel player two
        # 10 y vel player two
        # 11 angular vel player two
        # 12 x pos puck
        # 13 y pos puck
        # 14 x vel puck
        # 15 y vel puck
        # Keep Puck Mode
        # 16 time left player has puck
        # 17 time left other player has puck
        """
        # Extract values from observation
        x1, y1, theta1, vx1, vy1, omega1 = observation[:6]
        x2, y2, theta2, vx2, vy2, omega2 = observation[6:12]
        xp, yp, vxp, vyp = observation[12:16]
        keep_puck_mode, time_puck1, time_puck2 = observation[16:]
        
        # Compute relative positions
        x2_rel, y2_rel = x2 - x1, y2 - y1
        xp_rel, yp_rel = xp - x1, yp - y1
        
        # Compute absolute velocities (no transformation needed)
        absolute_velocities = [vx1, vy1, vx2, vy2, vxp, vyp]
        
        # Compute relative velocity components (parallel and orthogonal to puck direction)
        def velocity_components(vx, vy, px, py):
            """Computes velocity components parallel and orthogonal to the vector pointing to the puck."""
            puck_dir = np.array([px, py])
            puck_dir_norm = puck_dir / (np.linalg.norm(puck_dir) + 1e-8)  # Avoid division by zero
            v = np.array([vx, vy])
            v_parallel = np.dot(v, puck_dir_norm) * puck_dir_norm
            v_orthogonal = v - v_parallel
            return v_parallel.tolist(), v_orthogonal.tolist()
        
        v1_par, v1_ort = velocity_components(vx1, vy1, xp_rel, yp_rel)
        v2_par, v2_ort = velocity_components(vx2, vy2, xp_rel, yp_rel)
        
        # Flatten velocity components
        v1_par_x, v1_par_y = v1_par
        v1_ort_x, v1_ort_y = v1_ort
        v2_par_x, v2_par_y = v2_par
        v2_ort_x, v2_ort_y = v2_ort
        
        # Construct the new state vector
        enriched_observations = np.array([
            x1, y1,  # Absolute position of player 1
            x2_rel, y2_rel,  # Relative position of player 2
            xp, yp, xp_rel, yp_rel,  # Absolute and relative puck position
            *absolute_velocities,  # Absolute velocities of all objects
            v1_par_x, v1_par_y, v1_ort_x, v1_ort_y,  # Relative velocity components of player 1
            v2_par_x, v2_par_y, v2_ort_x, v2_ort_y,  # Relative velocity components of player 2
            keep_puck_mode, time_puck1, time_puck2  # Puck possession details
        ])
        
        return enriched_observations


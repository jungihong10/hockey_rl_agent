'''
DISCLAIMER: Template copied from https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/a35b1c1fa669428bf640a2c7101e66eb1627ac3a/gym_robotics/core.py#L8
'''

from abc import abstractmethod
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import error
# from Box2D.b2 import contactListener
from hockey.hockey_env import ContactDetector

from collections import deque
import math 

FPS = 50
SCALE = 60.0  # affects how fast-paced the game is, forces should be adjusted as well (Don't touch)

VIEWPORT_W = 600
VIEWPORT_H = 480
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CENTER_X = W / 2
CENTER_Y = H / 2
ZONE = W / 20
MAX_ANGLE = math.pi / 3  # Maximimal angle of racket
MAX_TIME_KEEP_PUCK = 15
GOAL_SIZE = 75

RACKETPOLY = [(-10, 20), (+5, 20), (+5, -20), (-10, -20), (-18, -10), (-21, 0), (-18, 10)]
RACKETFACTOR = 1.2

FORCEMULTIPLIER = 6000
SHOOTFORCEMULTIPLIER = 60
TORQUEMULTIPLIER = 400
MAX_PUCK_SPEED = 25



class GoalEnv(gym.Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


import hockey.hockey_env as h_env
from gymnasium import spaces
from gymnasium.spaces import Dict
import inspect
class HockeyHerEnv(h_env.HockeyEnv, GoalEnv):

    desired_goal = np.array(
        [CENTER_X + 245, CENTER_Y, 5, 0], # position puck
    )

    # we make 
    def __init__(self, env):

        self.is_parent_initialized = False
        super().__init__(env) # executes reset() method, function is overwritten by this child object, thus observation_space is still invalid! -> is_parent_initialized member

        self.observation_space = self.space() # change observation space here otherwise it is overwritten by parent env all the time

        self.is_parent_initialized = True
        self.reset() # crashes when we override reset as it is called in super().__init__


    def reset(self, one_starting=None, mode=None, seed=None, options=None):
 
        obs, info = super().reset(one_starting=one_starting, mode=mode, seed=seed, options=options) 
        # self.observation_space = self.space() # change observation space here otherwise it is overwritten by parent env all the time

        if self.is_parent_initialized:
            # Enforce that each GoalEnv uses a Goal-compatible observation space.
            if not isinstance(self.observation_space, gym.spaces.Dict):
                raise error.Error(
                    "GoalEnv requires an observation space of type gym.spaces.Dict"
                )
            for key in ["observation", "achieved_goal", "desired_goal"]:
                if key not in self.observation_space.spaces:
                    raise error.Error(
                        'GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(
                            key
                        )
                    )
                
            achieved_goal = np.array(
                [obs[12], obs[13], obs[14], obs[15]] # puck x,y, v
            )

            return {
                "observation": obs,
                "achieved_goal": achieved_goal,
                "desired_goal": self.desired_goal
            }, info
        
        else:
           return obs, info
        

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return {
            "observation": obs,
            "achieved_goal": np.array([obs[12], obs[13], obs[14], obs[15]]),
            "desired_goal": self.desired_goal
        }, reward, done, truncated, info

               
    # Puk x, y, velx, vely
    def space(self) -> spaces.Space:
        return spaces.Dict(
            dict(
                desired_goal = spaces.Box(low=np.array([0, 0, -10, -10]), high=np.array([W, H, 10, 10]), shape=(4,), dtype=np.float32), # puck position, agent position
                achieved_goal = spaces.Box(low=np.array([0, 0, -10, -10]), high=np.array([W, H, 10, 10]), shape=(4,), dtype=np.float32), # puck position, agent position
                observation = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)
            )
        )

    def compute_reward(self, achieved_goal, desired_goal, info):

        reward_weight = np.array([1/W, 1/H, 1/20, 1/20])
        # :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        p = 2
        r = -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                reward_weight,
            ),
            p)
        return r



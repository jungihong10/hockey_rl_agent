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

from enum import Enum
class Goal(Enum):
    goal = 0
    opp_goal = 1 # -1
    # pressing = 2 # opp moves back
    # bait = 2 # opp leaves goal
    ball_behind_opp = 2
    ball_behind_agent = 3 # -1
    side_fence_touched = 4
    agent_goal_fence_touched = 5
    opp_goal_fence_touched = 6
    # goalkeeper = 7 # opp shot defended
    # striker = 8 # shot attempt
    # opp_striker = 8 # shot attempt
    ball_possession = 7
    opp_ball_possession = 8
    VOID = 9



# contact detector to check if puck touches walls e.g. side/ goal fence
class PuckAgainstWallDetector(ContactDetector): # as only one contactListener is supported, we inherit
  '''
  given the 'world_objects' from the 'hockey_env' we can extract indices corresponding to wall objects:
  agent goal wall: {-5, -6}
  opponent goal wall: 
  side walls: 
  '''
  SIDE_WALL_IDX = [-5, -6]
  AGENT_GOAL_WALL_IDX = [-4, -3] # TODO are these correct?
  OPPONENT_GOAL_WALL_IDX = [-2, -1]


  def __init__(self, env, verbose=False):
    super().__init__(self)
    self.env = env
    self.verbose = verbose

  def isWorldObjectInContact(self, idx, contact, verbose=False, str_obj="wall object"):
    for id in idx:
      if self.env.world_objects[id] == contact.fixtureA.body or self.env.world_objects[id] == contact.fixtureB.body:
        if verbose:
           print(f'Puk hit {str_obj}.')
        return True
    return False
  
  def isSideWallInContact(self, contact):
     return self.isWorldObjectInContact(self.SIDE_WALL_IDX, contact, str_obj="side wall")

  def isAgentGoalWallInContact(self, contact):
     return self.isWorldObjectInContact(self.AGENT_GOAL_WALL_IDX, contact, str_obj="agent goal wall")
  
  def isOpponentGoalWallInContact(self, contact):
     return self.isWorldObjectInContact(self.OPPONENT_GOAL_WALL_IDX, contact, str_obj="opp goal wall")
     

  def BeginContact(self, contact):
    super().BeginContact(contact)
    if self.env.puck == contact.fixtureA.body or self.env.puck == contact.fixtureB.body:
      if self.isSideWallInContact(contact):
        self.env.side_wall_contact = True
      elif self.isOpponentGoalWallInContact(contact):
        self.env.opp_goal_wall_contact = True
      elif self.isAgentGoalWallInContact(contact):
        self.env.agent_goal_wall_contact = True 

  def EndContact(self, contact):
    super().EndContact(contact)
    if self.env.puck == contact.fixtureA.body or self.env.puck == contact.fixtureB.body:
      if self.isSideWallInContact(contact):
        self.env.side_wall_contact = False
      elif self.isOpponentGoalWallInContact(contact):
        self.env.opp_goal_wall_contact = False
      elif self.isAgentGoalWallInContact(contact):
        self.env.agent_goal_wall_contact = False 

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

    x_start_opp = None # reset every second (50 timesteps) # used for pressing goal


    # we make 
    def __init__(self, env):

        self.is_parent_initialized = False
        super().__init__(env) # executes reset() method, function is overwritten by this child object, thus observation_space is still invalid! -> is_parent_initialized member

        self.observation_space = self.space() # change observation space here otherwise it is overwritten by parent env all the time

        self.opponent_goal_wall_contact = False
        self.agent_goal_wall_contact = False
        self.side_wall_contact = False

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
                
            # override contact listener to fuse legacy contact listener with 'PuckAgainstWallDetector'
            self.world.contactListener = PuckAgainstWallDetector(self, verbose=self.verbose)

            # return spaces.Dict({
            #         desired_goal = obs,
            #         achieved_goal = None,
            #         observation= self.observation_space # TODO valid? # spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)
                
            # }), info
            #  self.current_state[key] = np.array([value], dtype=int)
            return {
                "observation": obs,
                "achieved_goal": Goal.VOID.value,
                "desired_goal": Goal.goal.value,
            }, info
        
        else:
           return obs, info
        

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
    
        return {
            "observation": obs,
            "achieved_goal": self.getAchievedGoal(obs).value,
            "desired_goal": Goal.goal.value, # will this be overwritten when goal sampling takes place?
        }, reward, done, truncated, info


    # class Goal(Enum):
    # goal = 0
    # opp_goal = 1 # -1
    # # pressing = 2 # opp moves back
    # bait = 2 # opp leaves goal
    # ball_behind_opp = 3
    # ball_behind_agent = 4 # -1
    # side_fence_touched = 5
    # agent_goal_fence_touched = 6
    # opp_goal_fence_touched = 6
    # goalkeeper = 7 # opp shot defended
    # striker = 8 # shot attempt
    # opp_striker = 8 # shot attempt
    # VOID = 10

    def getAchievedGoal(self, obs) -> Goal:
        achieved_goal = Goal.VOID
        if self.winner == 1:
           achieved_goal = Goal.goal
        elif self.winner == -1:
           achieved_goal = Goal.opp_goal
        elif self.side_wall_contact:
           achieved_goal = Goal.side_fence_touched
        elif self.agent_goal_wall_contact:
           achieved_goal = Goal.opp_goal_fence_touched
        elif self.opponent_goal_wall_contact:
           achieved_goal = Goal.agent_goal_fence_touched
        elif obs[6] < obs[12]: # puck behind opponent
           achieved_goal = Goal.ball_behind_opp
        elif obs[1] > obs[12]: # puck behind agent
           achieved_goal = Goal.ball_behind_agent
        if self.player1_has_puck:
            achieved_goal = Goal.ball_possession
        elif self.player2_has_puck:
            achieved_goal = Goal.opp_ball_possession
        # TODO goal striker, TODO goal bait
        return achieved_goal


    def space(self) -> spaces.Space:
        return spaces.Dict(
            dict(
                desired_goal = spaces.Discrete(len(Goal)),
                achieved_goal = spaces.Discrete(len(Goal)),
                observation= spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32) # self.observation_space # TODO valid? #
            )
        )

    def compute_reward(self, achieved_goal, desired_goal, info):

        # key_state_buffer = info['key_state_buffer']
        # TODO auswertung welches goal erreicht wurde, passiert wo genau???

        # Ensure achieved_goal and desired_goal are lists
        if not isinstance(achieved_goal, np.ndarray):
            achieved_goal = np.array([achieved_goal])
        if not isinstance(desired_goal, np.ndarray):
            desired_goal = np.array([desired_goal])

        rewards = []
        for ag, dg in zip(achieved_goal, desired_goal):
            # print(f"dg: {dg} \t ag: {ag}")
            # if dg == Goal.goal:
            #    print(f"it seems some resampling is taken place with goal {dg}")
            if dg == Goal.goal.value:
                reward = 100
            elif dg == Goal.opp_goal.value:
                reward = -100
            elif dg == Goal.ball_behind_agent.value:
                reward = -5
            elif dg == Goal.ball_behind_opp.value:
                reward = 5
            elif dg == Goal.agent_goal_fence_touched.value:
                reward = -2
            elif dg == Goal.opp_goal_fence_touched.value:
                # print("agent wall agent done")
                reward = 2
            elif dg == Goal.side_fence_touched.value:
                # print("agent wall agent done")
                reward = 1
            elif dg == Goal.ball_possession.value:
               reward = 1
            elif dg == Goal.opp_ball_possession.value:
               reward = -1
            else:
                reward = 0

            if ag == dg:
                rewards.append(reward)
            else:
                rewards.append(0)

        return np.array(rewards)


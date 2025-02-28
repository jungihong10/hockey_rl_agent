import hockey.hockey_env as h_env
from comprl.client import Agent
from ddpg import DDPG  # Assuming you're using DDPG
import numpy as np


class CustomHockeyAgent(Agent):
    """A custom hockey agent that loads a trained model."""

    def __init__(self, model_path: str):
        super().__init__()
        self.model = DDPG.load(model_path)
    
    def get_step(self, observation: list[float]) -> list[float]:
        action, _ = self.model.predict(np.array(observation))
        return action.tolist()

    def on_start_game(self, game_id) -> None:
        print("Custom agent game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"Game ended: {text_result} with score {stats[0]} - {stats[1]}")

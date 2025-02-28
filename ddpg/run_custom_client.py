from comprl.client import launch_client
from custom_agent import CustomHockeyAgent


def initialize_agent(agent_args):
    model_path = "models/ddpg_hockey_tournament.zip"  # Path to your trained model
    return CustomHockeyAgent(model_path)


def main():
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()

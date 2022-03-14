import os
import gym
from REINFORCE import REINFORCE
from config import Config
from pyvirtualdisplay import Display

if __name__ == "__main__":
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Init folder for model saves
    os.makedirs("models", exist_ok=True)

    # Init Gym env
    env = gym.make("CartPole-v1")

    # Init agent
    agent = REINFORCE(env)

    # Training params
    nb_episodes_per_epoch = Config.NB_EPISODES_PER_EPOCH
    nb_epoch = Config.NB_EPOCH
    nb_episodes_test = Config.NB_EPISODES_TEST

    # Train the agent
    agent.train(env, nb_episodes_per_epoch, nb_epoch)

    # Load best agent from training
    agent.load("best")

    # Evaluate and render the policy
    agent.test(env, nb_episodes=nb_episodes_test, render=True)

    display.stop()

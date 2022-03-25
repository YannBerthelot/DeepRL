import os
import gym
from n_step_A2C import A2C
from config import Config

if __name__ == "__main__":
    # Init folder for model saves
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    os.makedirs(Config.TENSORBOARD_PATH, exist_ok=True)

    # Init Gym env
    env = gym.make("CartPole-v1")

    # Init agent
    agent = A2C(env)

    # Training params
    nb_timesteps_train = Config.NB_TIMESTEPS_TRAIN
    nb_episodes_test = Config.NB_EPISODES_TEST

    # Train the agent
    agent.train(env, nb_timesteps_train)

    # Load best agent from training
    agent.load("best")

    # Evaluate and render the policy
    agent.test(env, nb_episodes=nb_episodes_test, render=True)

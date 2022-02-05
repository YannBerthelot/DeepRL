import os
import torch
import gym
from REINFORCE import REINFORCE
from config import Config

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    env = gym.make("CartPole-v1")
    agent = REINFORCE(env)
    nb_episodes_per_epoch = Config.NB_EPISODES_PER_EPOCH
    nb_epoch = Config.NB_EPOCH
    nb_episodes_test = Config.NB_EPISODES_TEST
    agent.train(env, nb_episodes_per_epoch, nb_epoch)
    agent.save()
    agent.test(env, nb_episodes=nb_episodes_test, render=True)
    agent.load("best")
    agent.test(env, nb_episodes=nb_episodes_test, render=True)

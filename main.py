import torch
import gym
from network import test_NN

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space.shape)
    test_NN(env)

import numpy
import gym


class Agent:
    """
    Base class for agents
    """

    def __init__(self) -> None:
        pass

    def select_action(self, observation):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self, env: gym.Env, nb_episodes: int, render: bool = False) -> None:
        raise NotImplementedError

    def save(self, name: str = "model"):
        raise NotImplementedError

    def load(self, name: str):
        raise NotImplementedError

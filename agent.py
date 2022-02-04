from libcst import Not
import numpy


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

    def loss(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

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

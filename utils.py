import numpy as np


class SimpleMinMaxScaler:
    def __init__(self, maxs: list, mins: list, feature_range: tuple = (0, 1)) -> None:
        self.max = np.array(maxs)
        self.min = np.array(mins)
        self.feature_range = feature_range

    def transform(self, x: np.array):
        return ((x - self.min) / (self.max - self.min)) * (
            self.feature_range[1] - self.feature_range[0]
        ) + self.feature_range[0]

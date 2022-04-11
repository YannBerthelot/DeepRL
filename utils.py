import numpy as np
import warnings
import torch
from network_utils import t


class SimpleMinMaxScaler:
    def __init__(self, maxs: list, mins: list, feature_range: tuple = (0, 1)) -> None:
        self.max = np.array(maxs)
        self.min = np.array(mins)
        self.feature_range = feature_range
        if feature_range[0] == feature_range[1]:
            raise ValueError(
                f"Feature range values must be different ({feature_range[0]} and {feature_range[1]})"
            )
        for i in range(len(self.max)):
            if self.max[i] <= self.min[i]:
                raise ValueError(
                    f"Mins must be inferior and different from max : Max {i} = {self.max[i]}, Min {i}={self.min[i]}"
                )

    def transform(self, x: np.array) -> np.array:
        return ((x - self.min) / (self.max - self.min)) * (
            self.feature_range[1] - self.feature_range[0]
        ) + self.feature_range[0]


class SimpleStandardizer:
    def __init__(self, clip=False, shift_mean=True, clipping_range=(-10, 10)) -> None:

        # Init internals
        self.count = 0
        self.mean = None
        self.M2 = None
        self.std = None

        # Fetch attributes
        self.shift_mean = shift_mean
        self.clip = clip
        self.clipping_range = clipping_range

    def partial_fit(self, newValue: np.array):
        # Welfor's online algorithm : https://en.m.wikipedia.org/wiki/Algorithms_for_calculating_variance
        self.count += 1
        if self.mean is None:
            self.mean = newValue
            self.std = np.zeros(len(newValue))
            self.M2 = np.zeros(len(newValue))
        delta = newValue - self.mean
        self.mean = self.mean + (delta / self.count)
        delta2 = newValue - self.mean
        self.M2 += np.multiply(delta, delta2)
        if self.count >= 2:
            self.std = np.sqrt(self.M2 / self.count)
            self.std = np.nan_to_num(self.std, nan=1)

    def transform(self, value: np.array):
        self.std[self.std == 0.0] = 1
        if self.shift_mean:
            new_value = (value - self.mean) / self.std
        else:
            new_value = value / self.std
        if self.clip:
            return np.clip(new_value, self.clipping_range[0], self.clipping_range[1])
        else:
            return new_value

    def pytorch_transform(self, value: torch.Tensor):
        self.std[self.std == 0.0] = 1
        if self.shift_mean:
            new_value = torch.div((torch.sub(value, t(self.mean))) / t(self.std))
        else:
            new_value = torch.div(value, t(self.std))
        if self.clip:
            return torch.clip(new_value, self.clipping_range[0], self.clipping_range[1])
        else:
            return new_value


def get_device(device_name: str) -> torch.DeviceObjType:
    """
    Chose the right device for PyTorch. If no GPU is available, it will use CPU.

    Args:
        device_name (str): The device to use between "GPU" and "CPU"

    Returns:
        torch.DeviceObjType: The Torch.Device to use
    """
    if device_name == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            warnings.warn("GPU not available, switching to CPU", UserWarning)
    else:
        device = torch.device("cpu")

    return device


class LinearSchedule:
    def __init__(self, start, end, t_max) -> None:
        self.start = start
        self.end = end
        self.t_max = t_max
        self.step = (start - end) / t_max

    def transform(self, t):
        return self.start - self.step * t

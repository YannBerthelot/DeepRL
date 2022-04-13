import numpy as np
import warnings
import torch
from network_utils import t


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

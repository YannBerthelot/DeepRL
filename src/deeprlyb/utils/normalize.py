import os
from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
import pathlib
import pickle
from deeprlyb.network.utils import t


class SimpleStandardizer:
    def __init__(
        self,
        clip: bool = False,
        shift_mean: bool = True,
        clipping_range: Tuple[int, int] = (-10, 10),
    ) -> None:

        # Init internals
        self._count = 0
        self.mean = None
        self.M2 = None
        self.std = None
        self._shape = None

        self.shift_mean = shift_mean
        self.clip = clip
        if clipping_range[0] > clipping_range[1]:
            raise ValueError(
                f"Lower clipping range ({clipping_range[0]}) is larger than High clipping range ({clipping_range[1]})"
            )
        elif clipping_range[0] == clipping_range[1]:
            raise ValueError(
                f"Lower clipping range ({clipping_range[0]}) is equal to High clipping range ({clipping_range[1]})"
            )
        else:
            self.clipping_range = clipping_range

    def partial_fit(self, newValue: npt.NDArray[np.float64]) -> None:
        # Welfor's online algorithm : https://en.m.wikipedia.org/wiki/Algorithms_for_calculating_variance
        self._count += 1
        if self.mean is None:
            self.mean = newValue
            self.std = np.zeros(len(newValue))
            self.M2 = np.zeros(len(newValue))
            self._shape = newValue.shape
        else:
            if self._shape != newValue.shape:
                raise ValueError(
                    f"The shape of samples has changed ({self._shape} to {newValue.shape})"
                )
        delta = newValue - self.mean
        self.mean = self.mean + (delta / self._count)
        delta2 = newValue - self.mean
        self.M2 += np.multiply(delta, delta2)
        if self._count >= 2:
            self.std = np.sqrt(self.M2 / self._count)
            self.std = np.nan_to_num(self.std, nan=1)

    @staticmethod
    def numpy_transform(
        value: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        shift_mean: bool = True,
        clip: bool = False,
        clipping_range: tuple = None,
    ) -> np.ndarray:
        if shift_mean:
            new_value = (value - mean) / std
        else:
            new_value = value / std
        if clip:
            return np.clip(new_value, clipping_range[0], clipping_range[1])
        else:
            return new_value

    @staticmethod
    def pytorch_transform(
        value: torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
        shift_mean: bool = True,
        clip: bool = False,
        clipping_range: tuple = None,
    ) -> torch.Tensor:

        if shift_mean:
            new_value = torch.div((torch.sub(value, t(mean))), t(std))
        else:
            new_value = torch.div(value, t(std))
        if clip:
            return torch.clip(new_value, clipping_range[0], clipping_range[1])
        else:
            return new_value

    def transform(
        self, value: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        std_temp = self.std
        std_temp[std_temp == 0.0] = 1
        if isinstance(value, np.ndarray):
            return self.numpy_transform(
                value,
                self.mean,
                self.std,
                self.shift_mean,
                self.clip,
                self.clipping_range,
            )
        elif isinstance(value, torch.Tensor):
            return self.pytorch_transform(
                value,
                self.mean,
                self.std,
                self.shift_mean,
                self.clip,
                self.clipping_range,
            )
        else:
            raise TypeError(f"type of transform input {type(value)} not handled atm")

    def save(
        self, path: Union[str, pathlib.Path] = ".", name: str = "standardizer"
    ) -> None:
        with open(os.path.join(path, name + ".pkl"), "wb") as file:
            pickle.dump(self, file)

    def load(self, path: Union[str, pathlib.Path], name: str = "standardizer") -> None:
        with open(os.path.join(path, name + ".pkl"), "rb") as file:
            save = pickle.load(file)
            self.std = save.std
            self.mean = save.mean
            self._count = save._count
            self.M2 = save.M2
            self._shape = save._shape
            self.shift_mean = save.shift_mean
            self.clip = save.clip
            self.clipping_range = save.clipping_range

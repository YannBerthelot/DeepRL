import torch
import torch.nn as nn
from typing import List
import warnings

# helper function to convert numpy arrays to tensors
def t(x):
    return torch.from_numpy(x).float()


def add_final_layer(activation, layers):
    if activation == "relu":
        layers.append(nn.ReLU())
    elif activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif activation == "tanh":
        layers.append(nn.Tanh())
    else:
        raise ValueError(f"Unrecognized activation function {activation}")
    return layers


def get_network_from_architecture(
    input_shape,
    output_shape,
    architecture: List[int],
    activation_function: str,
    mode: str = "actor",
    final_activation=None,
) -> torch.nn.modules.container.Sequential:
    """[summary]

    Args:
        architecture (List[int]): Architecture in terms of number of neurons per layer of the neural network

    Raises:
        ValueError: Returns an error if there aren't any layers provided

    Returns:
        torch.nn.modules.container.Sequential: The pytorch network
    """
    if activation_function == "relu":
        activation = nn.ReLU()
    elif activation_function == "tanh":
        activation = nn.Tanh()
    else:
        raise NotImplementedError

    if len(architecture) < 1:
        return nn.Linear(input_shape, output_shape)
    elif len(architecture) == 1:
        if mode == "common":
            return nn.Linear(input_shape, output_shape)
        elif mode == "critic":
            return nn.Sequential(
                activation,
                nn.Linear(input_shape, architecture[0]),
                activation,
                nn.Linear(architecture[0], output_shape),
            )
        elif mode == "actor":
            layers = [
                activation,
                nn.Linear(input_shape, architecture[0]),
                activation,
                nn.Linear(architecture[0], output_shape),
                nn.Softmax(dim=-1),
            ]
            if final_activation is not None:
                layers = add_final_layer(final_activation, layers)

            # layers[-2].weight.data.fill_(0.00)
            return nn.Sequential(*layers)

    else:
        layers = []
        for i, nb_neurons in enumerate(architecture):
            if i == 0:
                _input_shape = input_shape
                _output_shape = int(nb_neurons)
                layers.append(activation)
                layers.append(nn.Linear(_input_shape, _output_shape))
                layers.append(activation)
            else:
                _input_shape = int(architecture[i - 1])
                _output_shape = int(nb_neurons)
                layers.append(nn.Linear(_input_shape, _output_shape))
                layers.append(activation)
        _input_shape = architecture[-1]
        _output_shape = output_shape
        layers.append(nn.Linear(_input_shape, _output_shape))
        if mode == "actor":
            layers.append(nn.Softmax(dim=-1))
        # layers[-1].weight.data.fill_(0.00)
        if final_activation is not None:
            layers = add_final_layer(final_activation, layers)
        network = nn.Sequential(*layers)
        return network


def compute_KL_divergence(
    old_dist: torch.distributions.Distribution, dist: torch.distributions.Distribution
) -> float:
    if old_dist is not None:
        KL_divergence = torch.distributions.kl_divergence(old_dist, dist).mean()
    else:
        KL_divergence = 0
    return KL_divergence


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

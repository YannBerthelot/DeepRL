import torch
import torch.nn as nn
from typing import List

# helper function to convert numpy arrays to tensors
def t(x):
    return torch.from_numpy(x).float()


def get_network_from_architecture(
    input_shape,
    output_shape,
    architecture: List[int],
    activation_function: str,
    mode: str = "actor",
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
        else:
            return nn.Sequential(
                nn.Linear(input_shape, architecture[0]),
                activation,
                nn.Linear(architecture[0], output_shape),
            )
    else:
        layers = []
        for i, nb_neurons in enumerate(architecture):
            if i == 0:
                _input_shape = input_shape
                _output_shape = int(nb_neurons)
                print(_input_shape, _output_shape)
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
        network = nn.Sequential(*layers)
        return network

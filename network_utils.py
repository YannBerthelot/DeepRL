import torch
import torch.nn as nn
from typing import List
import numpy as np

# Read config
from config import Config


def get_network_from_architecture(
    input_shape, output_shape, architecture: List[int]
) -> torch.nn.modules.container.Sequential:
    """[summary]

    Args:
        architecture (List[int]): Architecture in terms of number of neurons per layer of the neural network

    Raises:
        ValueError: Returns an error if there aren't any layers provided

    Returns:
        torch.nn.modules.container.Sequential: The pytorch network
    """

    if len(architecture) < 1:
        raise ValueError("You need at least 1 layers")
    elif len(architecture) == 1:
        return nn.Sequential(
            nn.Linear(input_shape, architecture[0]),
            nn.ReLU(),
            nn.Linear(architecture[0], output_shape),
        )
    else:
        layers = []
        for i, nb_neurons in enumerate(architecture):
            if i == 0:
                _input_shape = input_shape
                _output_shape = nb_neurons
                layers.append(nn.Linear(_input_shape, _output_shape))
                layers.append(nn.ReLU())
            else:
                _input_shape = architecture[i - 1]
                _output_shape = nb_neurons
                layers.append(nn.Linear(_input_shape, _output_shape))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(Config.DROPOUT))
        _input_shape = architecture[-1]
        _output_shape = output_shape
        layers.append(nn.Linear(_input_shape, _output_shape))
        network = nn.Sequential(*layers)
        return network

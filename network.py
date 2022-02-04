from distutils.command import config
from platform import architecture
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import Config
from typing import List

if Config.DEVICE == "GPU":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        warnings.warn("GPU not available, switching to CPU", UserWarning)
else:
    device = torch.device("cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(PolicyNetwork, self).__init__()

        self.input_shape = observation_shape[0]
        self.output_shape = action_shape
        self.network = self.get_network_from_architecture(Config.NN_ARCHITECTURE)

    def get_network_from_architecture(
        self, architecture: List[int]
    ) -> torch.nn.modules.container.Sequential:
        """[summary]

        Args:
            architecture (List[int]): Architecture in terms of number of neurons per layer of the neural network

        Raises:
            ValueError: Returns an error if there aren't any layers provided

        Returns:
            torch.nn.modules.container.Sequential: The network
        """

        if len(architecture) < 1:
            raise ValueError("You need at least 1 layers")
        elif len(architecture) == 1:
            return nn.Sequential(
                nn.Linear(self.input_shape, architecture[0]),
                nn.ReLU(),
                nn.Linear(architecture[0], self.output_shape),
            )
        else:
            layers = []
            for i, nb_neurons in enumerate(architecture):
                if i == 0:
                    input_shape = self.input_shape
                    output_shape = nb_neurons
                    layers.append(nn.Linear(input_shape, output_shape))
                    layers.append(nn.ReLU())
                elif i == len(architecture) - 1:
                    input_shape = nb_neurons
                    output_shape = self.output_shape
                    layers.append(nn.Linear(input_shape, output_shape))
                else:
                    input_shape = architecture[i - 1]
                    output_shape = nb_neurons
                    layers.append(nn.Linear(input_shape, output_shape))
                    layers.append(nn.ReLU())
            network = nn.Sequential(*layers)
            return network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the policy pi(s, theta) for the given state s and for the current policy parameters theta

        Args:
            state (torch.Tensor): Torch tensor representation of the state

        Returns:
            torch.Tensor: Torch tensor representation of the actions
        """
        logits = self.network(state)
        pi_s = nn.Softmax(dim=0)(logits)
        return pi_s


def test_NN(env):
    print(f"Using {device} device")
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.n
    model = PolicyNetwork(obs_shape, action_shape).to(device)
    print(model)
    observation = env.reset()
    X = torch.tensor(observation, device=device)
    pred_probab = model(X)
    print(f"Predicted probas: {pred_probab}")

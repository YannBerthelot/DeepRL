from abc import abstractmethod
from typing import Tuple


# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import get_device

# Network creator tool
from network_utils import get_network_from_architecture

# Numpy
import numpy as np

ZERO = 1e-7


class BaseTorchAgent(nn.Module):
    def __init__(self, agent):
        super(BaseTorchAgent, self).__init__()
        self._agent = agent
        self.device = get_device(self.config["DEVICE"])

    @property
    def env(self):
        return self._agent.env

    @property
    def scaler(self):
        return self._agent.scaler

    @property
    def continuous(self):
        return self._agent.config["CONTINUOUS"]

    @property
    def config(self):
        return self._agent.config

    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def update_policy(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class ActorCriticRecurrentNetworks(nn.Module):
    """
    A class to represent the different PyTorch network of an A2C agent with recurrent networks

    ...

    Attributes
    ----------
    config : dict
        A dict containing all the config parameters of the agent
    action_dim :
        action dimension of the environment
    recurrent_layer : nn.LSTM
        The LSTM cell of the agent
    common_layers : nn.Sequential
        Fully connected layers before the LSTM cell
    actor_layers : nn.Sequential
        Fully connected layers after the LSTM cell and specific to the actor
    critic_layers : nn.Sequential
        Fully connected layers after the LSTM cell and specific to the critic
    device : str
        device used for PyTorch computations

    Methods
    -------
    forward(state, hidden):
        Shared layers of both actor and critic : LSTM and Fully Connected Layers (FCs)
    actor(state, hidden):
        Computes action probabilities for a given space
    critic(state, hidden):
        Evaluates a state
    """

    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, config):
        super(ActorCriticRecurrentNetworks, self).__init__()
        self.config = config
        self.action_dim = action_dim
        self.recurrent = self.config["RECURRENT"]
        self.continuous = self.config["CONTINUOUS"]
        common_architecture = eval(config["COMMON_NN_ARCHITECTURE"])
        actor_architecture = eval(config["ACTOR_NN_ARCHITECTURE"])
        critic_architecture = eval(config["CRITIC_NN_ARCHITECTURE"])

        # Device to run computations on
        self.device = "CPU"
        if self.config["CONTINUOUS"]:
            if self.config["LAW"] == "beta":
                final_activation = "relu"
            elif self.config["LAW"] == "normal":
                final_activation = "tanh"
        else:
            final_activation = None
        if self.recurrent:
            # LSTM cell(s)
            self._recurrent_layer = nn.LSTM(
                common_architecture[-1],
                hidden_dim,
                num_layers,
                batch_first=True,
            )

            # Layers before the LSTM
            self._common_layers = get_network_from_architecture(
                state_dim,
                common_architecture[-1],
                common_architecture,
                config["COMMON_ACTIVATION_FUNCTION"],
                mode="common",
            )

            # Layers after the LSTM specific to the actor
            self._actor_layers = get_network_from_architecture(
                hidden_dim,
                action_dim,
                actor_architecture,
                config["ACTOR_ACTIVATION_FUNCTION"],
                mode="actor",
                final_activation=final_activation,
            )
            # Layers after the LSTM specific to the critic
            self._critic_layers = get_network_from_architecture(
                hidden_dim,
                1,
                critic_architecture,
                config["CRITIC_ACTIVATION_FUNCTION"],
                mode="critic",
            )
        else:
            # Common layers
            self._common_layers = get_network_from_architecture(
                state_dim,
                common_architecture[-1],
                common_architecture,
                config["COMMON_ACTIVATION_FUNCTION"],
                mode="common",
            )

            # Layers after the LSTM specific to the actor
            self._actor_layers = get_network_from_architecture(
                common_architecture[-1],
                action_dim,
                actor_architecture,
                config["ACTOR_ACTIVATION_FUNCTION"],
                mode="actor",
            )
            # Layers after the LSTM specific to the critic
            self._critic_layers = get_network_from_architecture(
                common_architecture[-1],
                1,
                critic_architecture,
                config["CRITIC_ACTIVATION_FUNCTION"],
                mode="critic",
            )
        if self.config["CONTINUOUS"]:
            logstds_param = nn.Parameter(torch.full((action_dim,), 0.1))
            self.register_parameter("logstds", logstds_param)

    def LSTM(self, x, hidden):
        # x = torch.relu(x)
        x = x.view(
            -1,
            1,
            eval(self.config["COMMON_NN_ARCHITECTURE"])[-1],
        )

        x, hidden = self._recurrent_layer(x, hidden)
        return x, hidden

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Layers shared by the actor and the critic:
        -Some FCs
        -LSTM cell

        Args:
            state (torch.Tensor): State to be processed
            hidden (torch.Tensor): Hidden state of the LSTM

        Returns:
            Tuple[Torch.Tensor, Torch.Tensor]: The processed state and the new hidden state of the LSTM
        """

        return self._common_layers(state)

    def actor(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the action probabilities for the given state and the current actor parameters

        Args:
            state (torch.Tensor): The state for which to compute action probabilities
            hidden (torch.Tensor): The current hiden state of the LSTM

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the action probabilities and the new hidden state
        """
        # with torch.no_grad():
        # embedded_obs, hidden = self.forward(state, hidden)
        # hidden = (hidden[0].detach(), hidden[1].detach())
        x = self._actor_layers(state)
        if self.config["CONTINUOUS"]:
            stds = torch.clamp(self.logstds.exp(), 1e-3, 0.1)
            if self.config["LAW"] == "normal":
                return (torch.distributions.Normal(x, stds),)
            elif self.config["LAW"] == "beta":
                return torch.distributions.Beta(x + ZERO, stds + ZERO)
            else:
                raise ValueError(f'Unknown law {self.config["LAW"]}')
        else:
            dist = torch.distributions.Categorical(probs=x)
            return dist

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns the value estimation for the given state and the current critic parameters

        Args:
            state (torch.Tensor): The state for which to compute the value
            hidden (torch.Tensor): The current hiden state of the LSTM

        Returns:
            torch.Tensor: A tuple containing the state value and the new hidden state
        """
        # with torch.no_grad():
        # embedded_obs, _ = self.forward(state, hidden)
        value = self._critic_layers(state)
        return value.view(1)

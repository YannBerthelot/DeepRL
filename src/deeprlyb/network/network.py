from abc import abstractmethod
from typing import Tuple, Dict


# PyTorch
import torch
import torch.nn as nn

# Network creator tool
from deeprlyb.network.utils import get_network_from_architecture, get_device


ZERO = 1e-7


class BaseTorchAgent(nn.Module):
    def __init__(self, agent):
        super(BaseTorchAgent, self).__init__()
        self._agent = agent
        self.device = get_device(self.config["HARDWARE"]["device"])

    @property
    def env(self):
        return self._agent.env

    @property
    def scaler(self):
        return self._agent.scaler

    @property
    def continuous(self):
        return self._agent.config["GLOBAL"].getboolean("continuous")

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
    TBRD
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        architecture,
        actor=True,
    ):
        super(ActorCriticRecurrentNetworks, self).__init__()
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._architecture = architecture[1:-1].split(",")
        self.actor = actor
        self.network = self.init_layers()

    @property
    def architecture(self):
        return self._architecture

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def num_layers(self):
        return self._num_layers

    def initialize_hidden_states(self):
        hiddens = {}
        for i, layer in enumerate(self.network):
            if isinstance(layer, torch.nn.modules.rnn.LSTM):
                hiddens[i] = ActorCriticRecurrentNetworks.get_initial_states(
                    hidden_size=layer.hidden_size, num_layers=layer.num_layers
                )
        return hiddens

    def init_layers(self) -> torch.nn.Sequential:
        # Device to run computations on
        self.device = "CPU"
        output_size = self.action_dim if self.actor else 1
        return get_network_from_architecture(
            self.state_dim,
            output_size,
            self.architecture,
            activation_function="relu",
            mode="actor",
        )

    def forward(
        self,
        input: torch.Tensor,
        hiddens: dict = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Layers shared by the actor and the critic:
        -Some FCs
        -LSTM cell

        Args:
            state (torch.Tensor): State to be processed
            hidden (Dict[torch.Tensor]): Hidden states of the LSTM

        Returns:
            Tuple[Torch.Tensor, Torch.Tensor]: The processed state and the new hidden state of the LSTM
        """
        for i, layer in enumerate(self.network):
            if isinstance(layer, torch.nn.modules.rnn.LSTM):
                input = input.view(-1, 1, layer.input_size)
                input, hiddens[i] = layer(input, hiddens[i])
            else:
                input = layer(input)
        return input, hiddens

    def get_initial_states(hidden_size, num_layers):
        h_0, c_0 = None, None

        h_0 = torch.zeros(
            (
                num_layers,
                1,
                hidden_size,
            ),
            dtype=torch.float,
        )

        c_0 = torch.zeros(
            (
                num_layers,
                1,
                hidden_size,
            ),
            dtype=torch.float,
        )
        return (h_0, c_0)

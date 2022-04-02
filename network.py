from typing import Tuple
import warnings

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Network creator tool
from network_utils import get_network_from_architecture, t

# Numpy
import numpy as np

import wandb


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
        common_architecture = eval(config["COMMON_NN_ARCHITECTURE"])
        actor_architecture = eval(config["ACTOR_NN_ARCHITECTURE"])
        critic_architecture = eval(config["CRITIC_NN_ARCHITECTURE"])

        # Device to run computations on
        self.device = "CPU"

        if self.recurrent:
            # LSTM cell(s)
            self.recurrent_layer = nn.LSTM(
                common_architecture[-1],
                hidden_dim,
                num_layers,
                batch_first=True,
            )

            # Layers before the LSTM
            self.common_layers = get_network_from_architecture(
                state_dim,
                common_architecture[-1],
                common_architecture,
                config["COMMON_ACTIVATION_FUNCTION"],
                mode="common",
            )

            # Layers after the LSTM specific to the actor
            self.actor_layers = get_network_from_architecture(
                config["HIDDEN_SIZE"],
                action_dim,
                actor_architecture,
                config["ACTOR_ACTIVATION_FUNCTION"],
                mode="actor",
            )
            # Layers after the LSTM specific to the critic
            self.critic_layers = get_network_from_architecture(
                config["HIDDEN_SIZE"],
                1,
                critic_architecture,
                config["CRITIC_ACTIVATION_FUNCTION"],
                mode="critic",
            )
        else:
            # Common layers
            self.common_layers = get_network_from_architecture(
                state_dim,
                common_architecture[-1],
                common_architecture,
                config["COMMON_ACTIVATION_FUNCTION"],
                mode="common",
            )

            # Layers after the LSTM specific to the actor
            self.actor_layers = get_network_from_architecture(
                common_architecture[-1],
                action_dim,
                actor_architecture,
                config["ACTOR_ACTIVATION_FUNCTION"],
            )
            # Layers after the LSTM specific to the critic
            self.critic_layers = get_network_from_architecture(
                common_architecture[-1],
                1,
                critic_architecture,
                config["CRITIC_ACTIVATION_FUNCTION"],
            )

    def forward(
        self, state: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        x = F.relu(self.common_layers(state))
        if self.recurrent:
            x = x.view(-1, 1, eval(self.config["COMMON_NN_ARCHITECTURE"])[-1])
            x, lstm_hidden = self.recurrent_layer(x, hidden)
        return x, hidden

    def actor(
        self, state: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the action probabilities for the given state and the current actor parameters

        Args:
            state (torch.Tensor): The state for which to compute action probabilities
            hidden (torch.Tensor): The current hiden state of the LSTM

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the action probabilities and the new hidden state
        """
        x, hidden = self.forward(state, hidden)
        x = self.actor_layers(x)
        probs = F.softmax(x, dim=2)
        return probs, hidden

    def critic(self, state: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Returns the value estimation for the given state and the current critic parameters

        Args:
            state (torch.Tensor): The state for which to compute the value
            hidden (torch.Tensor): The current hiden state of the LSTM

        Returns:
            torch.Tensor: A tuple containing the state value and the new hidden state
        """
        x, hidden = self.forward(state, hidden)
        values = self.critic_layers(x)
        return values


class ActorCritic(nn.Module):
    def __init__(self, observation_shape, action_shape, config):
        super(ActorCritic, self).__init__()
        self.config = config
        # Set the Torch device
        if config["DEVICE"] == "GPU":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available():
                warnings.warn("GPU not available, switching to CPU", UserWarning)
        else:
            self.device = torch.device("cpu")
        # Create the network architecture given the observation, action and config shapes
        self.actorcritic = ActorCriticRecurrentNetworks(
            observation_shape[0],
            action_shape,
            self.config["HIDDEN_SIZE"],
            self.config["HIDDEN_LAYERS"],
            self.config,
        )
        print(self.actorcritic.critic_layers)
        self.actorcritic_target = ActorCriticRecurrentNetworks(
            observation_shape[0],
            action_shape,
            self.config["HIDDEN_SIZE"],
            self.config["HIDDEN_LAYERS"],
            self.config,
        )
        self.actorcritic_target.load_state_dict(self.actorcritic.state_dict())
        self.actorcritic_target.eval()
        print(self.actorcritic)
        if self.config["logging"] == "wandb":
            wandb.watch(self.actorcritic)

        # Optimize to use for weight update (SGD seems to work poorly, switching to RMSProp) given our learning rate
        self.optimizer = optim.Adam(
            self.actorcritic.parameters(),
            lr=config["LEARNING_RATE"],
        )

        # Init stuff
        self.loss = None
        self.epoch = 0
        self.writer = None
        self.index = 0
        self.hidden = None
        self.old_probs = None
        self.KLdiv = nn.KLDivLoss(reduction="batchmean")

    def select_action(self, observation: np.array, hidden: np.array) -> np.array:
        """Select the action based on the observation and the current parametrized policy

        Args:
            observation (np.array): The current state observation

        Returns:
            np.array : The selected action(s)
        """
        with torch.no_grad():
            observation = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)[
                :, None, :
            ]
            probs, hidden = self.actorcritic_target.actor(observation, hidden)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        return action.detach().data.numpy()[0, 0], hidden

    def update_policy(
        self,
        state: np.array,
        action: np.array,
        n_step_return: np.array,
        next_state: np.array,
        hidden: np.array,
        next_hidden: np.array,
        done: bool = False,
    ) -> None:
        """
        Update the policy's parameters according to the n-step A2C updates rules. see : https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b


        Args:
            state (np.array): Observation of the state
            action (np.array): The selected action
            n_step_return (np.array): The n-step return
            next_state (np.array): The state n-step after state
            done (bool, optional): Wether the episode if finished or not at next-state. Used to handle 1-step. Defaults to False.
        """

        # For logging purposes
        self.index += 1

        ## Compute the losses
        # Actor loss
        state = t(state).reshape(1, -1)[:, None, :]
        next_state = t(next_state).reshape(1, -1)[:, None, :]
        probs, _ = self.actorcritic.actor(
            state,
            hidden,
        )
        if self.old_probs is not None:
            KL_divergence = self.KLdiv(self.old_probs, probs)
        else:
            KL_divergence = 0
        self.old_probs = probs
        dist = torch.distributions.Categorical(probs=probs)

        # Entropy loss
        entropy_loss = -dist.entropy()

        # Critic loss
        advantage = (
            n_step_return
            + (1 - done)
            * self.config["GAMMA"] ** self.config["N_STEPS"]
            * self.actorcritic.critic(next_state, next_hidden)
            - self.actorcritic.critic(state, hidden)
        )

        # Update critic
        critic_loss = advantage.pow(2).mean()

        # Update actor
        actor_loss = -dist.log_prob(t(np.array([action]))) * advantage.detach()

        loss = (
            actor_loss
            + self.config["VALUE_FACTOR"] * critic_loss
            + self.config["ENTROPY_FACTOR"] * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.mean().backward(retain_graph=True)
        self.optimizer.step()
        if self.index % self.config["TARGET_UPDATE"] == 0:
            self.actorcritic_target.load_state_dict(self.actorcritic.state_dict())
        # Logging
        if self.writer:
            if self.config["logging"] == "tensorboard":
                self.writer.add_scalar("Train/entropy loss", -entropy_loss, self.index)
                self.writer.add_scalar("Train/policy loss", actor_loss, self.index)
                self.writer.add_scalar("Train/critic loss", critic_loss, self.index)
                self.writer.add_scalar("Train/total loss", loss, self.index)
                self.writer.add_scalar("Train/kl divergence", KL_divergence, self.index)
        else:
            warnings.warn("No Tensorboard writer available")
        if self.config["logging"] == "wandb":
            wandb.log(
                {
                    "Train/entropy loss": -entropy_loss,
                    "Train/actor loss": actor_loss,
                    "Train/critic loss": critic_loss,
                    "Train/total loss": loss,
                    "Train/KL divergence": KL_divergence,
                },
                commit=False,
            )

    def get_action_probabilities(self, state: np.array) -> np.array:
        """
        Computes the policy pi(s, theta) for the given state s and for the current policy parameters theta.
        Same as forward method with a clearer name for teaching purposes, but as forward is a native method that needs to exist we keep both.
        Additionnaly this methods outputs np.array instead of torch.Tensor to prevent the existence of pytorch stuff outside of network.py

        Args:
            state (np.array): np.array representation of the state

        Returns:
            np.array: np.array representation of the action probabilities
        """
        return self.actor(t(state)).detach().cpu().numpy()

    def get_value(self, state: np.array) -> np.array:
        """
        Computes the state value for the given state s and for the current policy parameters theta.
        Same as forward method with a clearer name for teaching purposes, but as forward is a native method that needs to exist we keep both.
        Additionnaly this methods outputs np.array instead of torch.Tensor to prevent the existence of pytorch stuff outside of network.py

        Args:
            state (np.array): np.array representation of the state

        Returns:
            np.array: np.array representation of the action probabilities
        """
        return self.critic(t(state)).detach().cpu().numpy()

    def save(self, name: str = "model"):
        """
        Save the current model

        Args:
            name (str, optional): [Name of the model]. Defaults to "model".
        """
        torch.save(self.actorcritic, f'{self.config["MODEL_PATH"]}/{name}.pth')

    def load(self, name: str = "model"):
        """
        Load the designated model

        Args:
            name (str, optional): The model to be loaded (it should be in the "models" folder). Defaults to "model".
        """
        print("Loading")
        self.actorcritic = torch.load(f'{self.config["MODEL_PATH"]}/{name}.pth')
        self.actorcritic_target = torch.load(f'{self.config["MODEL_PATH"]}/{name}.pth')

    def get_initial_states(self):
        h_0, c_0 = None, None

        h_0 = torch.zeros(
            (
                self.actorcritic.recurrent_layer.num_layers,
                1,
                self.actorcritic.recurrent_layer.hidden_size,
            ),
            dtype=torch.float,
        )
        h_0 = h_0.to(device=self.device)

        c_0 = torch.zeros(
            (
                self.actorcritic.recurrent_layer.num_layers,
                1,
                self.actorcritic.recurrent_layer.hidden_size,
            ),
            dtype=torch.float,
        )
        c_0 = c_0.to(device=self.device)
        return (h_0, c_0)

from typing import Tuple
import warnings

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from buffer import RolloutBuffer

# Network creator tool
from network_utils import get_network_from_architecture, t, compute_KL_divergence
from utils import SimpleStandardizer, get_device, LinearSchedule

# Numpy
import numpy as np

import wandb

ZERO = 1e-7


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
                common_architecture[-1],
                action_dim,
                actor_architecture,
                config["ACTOR_ACTIVATION_FUNCTION"],
                mode="actor",
                final_activation=final_activation,
            )
            # Layers after the LSTM specific to the critic
            self.critic_layers = get_network_from_architecture(
                common_architecture[-1],
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
                mode="actor",
            )
            # Layers after the LSTM specific to the critic
            self.critic_layers = get_network_from_architecture(
                common_architecture[-1],
                1,
                critic_architecture,
                config["CRITIC_ACTIVATION_FUNCTION"],
                mode="critic",
            )
        if self.config["CONTINUOUS"]:
            logstds_param = nn.Parameter(torch.full((action_dim,), 0.1))
            self.register_parameter("logstds", logstds_param)

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

        x = self.common_layers(state)
        if self.recurrent:
            # If recurrent, we need to add an activaction function there, otherwise it's done in the actor or critic networks
            # x = torch.relu(x)
            x = x.view(-1, 1, eval(self.config["COMMON_NN_ARCHITECTURE"])[-1])
            x, hidden = self.recurrent_layer(x, hidden)
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
        # with torch.no_grad():
        # embedded_obs, hidden = self.forward(state, hidden)
        # hidden = (hidden[0].detach(), hidden[1].detach())
        x = self.actor_layers(state)
        if self.config["CONTINUOUS"]:
            stds = torch.clamp(self.logstds.exp(), 1e-3, 0.1)
            if self.config["LAW"] == "normal":
                return (
                    torch.distributions.Normal(x, stds),
                    hidden,
                )
            elif self.config["LAW"] == "beta":
                return torch.distributions.Beta(x + ZERO, stds + ZERO), hidden
            else:
                raise ValueError(f'Unknown law {self.config["LAW"]}')
        else:
            dist = torch.distributions.Categorical(probs=x)
            return dist, hidden

    def critic(self, state: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
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
        value = self.critic_layers(state)
        return value.reshape(-1, 1)


class ActorCritic(nn.Module):
    def __init__(self, observation_shape, action_shape, config, scaler=None, env=None):
        super(ActorCritic, self).__init__()
        self.config = config
        self.scaler = scaler
        self.env = env
        self.continuous = config["CONTINUOUS"]
        # Set the Torch device

        self.device = get_device(config["DEVICE"])
        # Create the network architecture given the observation, action and config shapes
        self.actorcritic = ActorCriticRecurrentNetworks(
            observation_shape[0],
            action_shape,
            self.config["HIDDEN_SIZE"],
            self.config["HIDDEN_LAYERS"],
            self.config,
        )
        print(self.actorcritic)

        if self.config["logging"] == "wandb":
            wandb.watch(self.actorcritic)

        # Optimize to use for weight update (SGD seems to work poorly, switching to RMSProp) given our learning rate
        self.optimizer = optim.Adam(
            self.actorcritic.parameters(),
            lr=config["LEARNING_RATE"],
        )

        self.target_var_scaler = SimpleStandardizer()
        self.advantages_var_scaler = SimpleStandardizer()
        self.lr_scheduler = LinearSchedule(
            self.config["LEARNING_RATE"],
            self.config["LEARNING_RATE_END"],
            self.config["NB_TIMESTEPS_TRAIN"],
        )
        # Init stuff
        self.loss = None
        self.epoch = 0
        self.writer = None
        self.index = 0
        self.hidden = None
        self.old_probs = None
        self.old_dist = None
        self.KLdiv = nn.KLDivLoss(reduction="batchmean")
        self.advantages = []
        self.targets = []

    def select_action(self, observation: np.array, hidden: np.array) -> np.array:
        """Select the action based on the observation and the current parametrized policy

        Args:
            observation (np.array): The current state observation

        Returns:
            np.array : The selected action(s)
        """

        observation = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)[
            :, None, :
        ]
        embedded_observation, new_hidden = self.actorcritic.forward(observation, hidden)
        dist, _ = self.actorcritic.actor(embedded_observation, hidden)
        action = dist.sample()
        if self.continuous:
            if self.config["LAW"] == "normal":
                # action = (
                #     action
                #     * (self.env.action_space.high - self.env.action_space.low)
                #     + self.env.action_space.low
                # )
                action = np.clip(
                    action, self.env.action_space.low, self.env.action_space.high
                )
            elif self.config["LAW"] == "beta":
                action = (
                    action * (self.env.action_space.high - self.env.action_space.low)
                    + self.env.action_space.low
                )
            return action.detach().data.numpy()[0][0], hidden
        else:
            action = action.reshape(-1, 1)
            return action.item(), new_hidden

    def update_policy(
        self,
        rollout: RolloutBuffer,
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

        self.optimizer = optim.Adam(
            self.actorcritic.parameters(),
            lr=self.lr_scheduler.transform(self.index),
        )
        if self.config["RECURRENT"]:
            embedded_obs, hidden = self.actorcritic.forward(
                t(rollout.states), t(rollout.hiddens)
            )
            embedded_next_obs, next_hidden = self.actorcritic.forward(
                t(rollout.next_states), t(rollout.next_hiddens)
            )
        else:
            embedded_obs, hidden = self.actorcritic.forward(t(rollout.states), None)
            embedded_next_obs, next_hidden = self.actorcritic.forward(
                t(rollout.next_states), None
            )
        dist, _ = self.actorcritic.actor(
            embedded_obs,
            hidden,
        )

        log_probs = dist.log_prob(t(rollout.actions))

        empirical_return = t(rollout.returns) + (1 - t(rollout.dones)) * self.config[
            "GAMMA"
        ] ** self.config["N_STEPS"] * self.actorcritic.critic(
            embedded_next_obs, next_hidden
        )
        estimated_return = self.actorcritic.critic(embedded_obs, hidden)

        # Target scaling
        if self.scaler is not None:
            self.scaler.partial_fit(empirical_return.detach().numpy().flatten())
        if (
            (self.scaler is not None)
            and (self.config["TARGET_SCALING"])
            and (self.index > 2)
            and self.index <= self.config["LEARNING_START"]
        ):
            empirical_return = self.scaler.pytorch_transform(empirical_return)

        advantages = empirical_return - estimated_return
        # with torch.no_grad():
        #     KL_divergence = np.exp(compute_KL_divergence(self.old_dist, dist)) - 1

        # Losses (be careful that all its components are torch tensors with grad on)
        critic_loss = advantages.pow(2).mean()
        actor_loss = (-log_probs * advantages).mean()
        entropy_loss = -dist.entropy().mean()
        kl_loss = 0  # -KL_divergence

        loss = (
            actor_loss
            + self.config["VALUE_FACTOR"] * critic_loss
            + self.config["ENTROPY_FACTOR"] * entropy_loss
            + self.config["KL_FACTOR"] * kl_loss
        )
        if self.index > self.config["LEARNING_START"]:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=False)
            self.gradient_clipping()
            self.optimizer.step()

        # KPIs
        # explained_variance = self.compute_explained_variance(
        #     empirical_return.detach().numpy().flatten(),
        #     advantage.detach().numpy().flatten(),
        # )
        explaied_variance = 0
        self.old_dist = dist

        # Logging
        if self.writer:
            if self.config["logging"] == "tensorboard":
                self.writer.add_scalar("Train/entropy loss", -entropy_loss, self.index)
                self.writer.add_scalar(
                    "Train/leaarning rate",
                    self.lr_scheduler.transform(self.index),
                    self.index,
                )
                self.writer.add_scalar("Train/policy loss", actor_loss, self.index)
                self.writer.add_scalar("Train/critic loss", critic_loss, self.index)
                self.writer.add_scalar("Train/total loss", loss, self.index)
                # self.writer.add_scalar(
                #     "Train/explained variance", explained_variance, self.index
                # )
                # self.writer.add_scalar("Train/kl divergence", KL_divergence, self.index)
        else:
            warnings.warn("No Tensorboard writer available")
        if self.config["logging"] == "wandb":
            wandb.log(
                {
                    "Train/entropy loss": -entropy_loss,
                    "Train/actor loss": actor_loss,
                    "Train/critic loss": critic_loss,
                    "Train/total loss": loss,
                    # "Train/explained variance": explained_variance,
                    # "Train/KL divergence": KL_divergence,
                    "Train/learning rate": self.lr_scheduler.transform(self.index),
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
        self.actorcritic.config = self.config

        print(self.actorcritic)

    def get_initial_states(self):
        if self.config["RECURRENT"]:
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
        else:
            return None

    def fit_transform(self, input):
        self.scaler.partial_fit(input)
        if self.index > 2:
            return t(self.scaler.transform(input))

    def gradient_clipping(self):
        nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            self.config["GRADIENT_CLIPPING"],
        )  # gradient clipping

    def compute_explained_variance(self, target, advantage):
        self.target_var_scaler.partial_fit(target)
        self.advantages_var_scaler.partial_fit(advantage)

        var_targets = self.target_var_scaler.std
        var_advantages = self.advantages_var_scaler.std

        if var_targets == 0:
            explained_variance = 0
        else:
            explained_variance = 1 - var_advantages / var_targets

        return explained_variance

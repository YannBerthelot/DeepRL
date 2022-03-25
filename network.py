from distutils.log import warn
import gym
import warnings

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Network creator tool
from network_utils import get_network_from_architecture, t

# Numpy
import numpy as np

# Read config
from config import Config

# Set the Torch device
if Config.DEVICE == "GPU":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        warnings.warn("GPU not available, switching to CPU", UserWarning)
else:
    device = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()

        # Create the network architecture given the observation, action and config shapes
        self.input_shape = observation_shape
        self.output_shape = action_shape
        self.model = get_network_from_architecture(
            self.input_shape,
            self.output_shape,
            Config.ACTOR_NN_ARCHITECTURE,
            Config.ACTOR_ACTIVATION_FUNCTION,
            mode="actor",
        )

    def forward(self, X):
        return self.model(X)


class Critic(nn.Module):
    def __init__(self, observation_shape):
        super().__init__()

        # Create the network architecture given the observation, action and config shapes
        self.input_shape = observation_shape
        self.output_shape = 1
        self.model = get_network_from_architecture(
            self.input_shape,
            self.output_shape,
            Config.CRITIC_NN_ARCHITECTURE,
            Config.CRITIC_ACTIVATION_FUNCTION,
            mode="critic",
        )

    def forward(self, X):
        return self.model(X)


class ActorCritic(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(ActorCritic, self).__init__()

        # Create the network architecture given the observation, action and config shapes
        self.actor = Actor(observation_shape[0], action_shape)
        self.critic = Critic(observation_shape[0])

        # Optimize to use for weight update (SGD seems to work poorly, switching to RMSProp) given our learning rate
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=Config.ACTOR_LEARNING_RATE,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=Config.CRITIC_LEARNING_RATE,
        )
        # Init stuff
        self.loss = None
        self.epoch = 0
        self.writer = None
        self.index = 0

    def select_action(self, observation: np.array) -> np.array:
        """Select the action based on the observation and the current parametrized policy

        Args:
            observation (np.array): The current state observation

        Returns:
            np.array : The selected action(s)
        """
        probs = self.actor(t(observation))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action.detach().data.numpy()

    def update_policy(
        self,
        state: np.array,
        action: np.array,
        n_step_return: np.array,
        next_state: np.array,
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
        probs = self.actor(t(state))
        dist = torch.distributions.Categorical(probs=probs)

        # Entropy loss
        entropy_loss = -dist.entropy()

        # Critic loss
        advantage = (
            n_step_return
            + (1 - done) * Config.GAMMA ** Config.N_STEPS * self.critic(t(next_state))
            - self.critic(t(state))
        )

        # Update critic
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -dist.log_prob(t(action)) * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Logging
        if self.writer:
            self.writer.add_scalar("Loss/entropy", entropy_loss, self.index)
            self.writer.add_scalar("Loss/policy", actor_loss, self.index)
            self.writer.add_scalar("Loss/critic", critic_loss, self.index)
        else:
            warnings.warn("No Tensorboard writer available")

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
        torch.save(self.actor, f"{Config.MODEL_PATH}/{name}_actor.pth")
        torch.save(self.critic, f"{Config.MODEL_PATH}/{name}_critic.pth")

    def load(self, name: str = "model"):
        """
        Load the designated model

        Args:
            name (str, optional): The model to be loaded (it should be in the "models" folder). Defaults to "model".
        """
        print("Loading")
        self.actor = torch.load(f"{Config.MODEL_PATH}/{name}_actor.pth")
        self.critic = torch.load(f"{Config.MODEL_PATH}/{name}_critic.pth")


def test_NN(env):
    print(f"Using {device} device")
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.n
    model = Network(obs_shape, action_shape).to(device)

    observation = env.reset()
    state = np.array(observation)
    reward = np.array([1.0])
    action = [1]
    next_state = np.array(observation)
    done = False
    model.update_policy(state, action, reward, next_state, done)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    test_NN(env)

import gym
import warnings

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Network creator tool
from network_utils import get_network_from_architecture

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


class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(PolicyNetwork, self).__init__()

        # Create the network architecture given the observation, action and config shapes
        self.input_shape = observation_shape[0]
        self.output_shape = action_shape
        self.policy_network = get_network_from_architecture(
            self.input_shape, self.output_shape, Config.POLICY_NN_ARCHITECTURE
        )

        # Optimize to use for weight update (SGD seems to work poorly, switching to RMSProp) given our learning rate
        self.optimizer = optim.RMSprop(
            self.policy_network.parameters(),
            lr=Config.POLICY_LEARNING_RATE,
        )

        # Init stuff
        self.loss = None
        self.epoch = 0
        self.writer = None
        self.index = 0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the policy pi(s, theta) for the given state s and for the current policy parameters theta

        Args:
            state (torch.Tensor): Torch tensor representation of the state

        Returns:
            torch.Tensor: Torch tensor representation of the action probabilities
        """
        # Forward pass of the state representation through the network to get logits
        logits = self.policy_network(state)

        # Softmax over the logits to get action probabilities
        pi_s = nn.Softmax(dim=0)(logits)

        # For logging purposes
        self.index += 1

        return pi_s

    def get_action_probabilities(self, state: np.array) -> np.array:
        """
        Computes the policy pi(s, theta) for the given state s and for the current policy parameters theta.
        Same as forward method with a clearer name for teaching purposes, but as forward is a native method that needs to exist we keep both.
        Additionnaly this methods outputs np.array instead of torch.Tensor to prevent the existence of pytorch stuff outside of network.py

        Args:
            state (np.array): np.array representation of the state

        Returns:
            torch.Tensor: np.array representation of the action probabilities
        """
        state = torch.Tensor(state, device=device)
        logits = self.policy_network(state)
        pi_s = nn.Softmax(dim=0)(logits)
        return pi_s.detach().cpu().numpy()

    def update_policy(self, state: np.array, action: np.array, G: np.array) -> None:
        """
        Update the policy using the given state, action and reward and the REINFORCE update rule

        Args:
            state (np.array): The state to consider
            action (np.array): The action to consider
            G (np.array): The discounted return to consider
        """
        state = torch.from_numpy(state).to(device=device)
        action = torch.as_tensor(action).to(device=device)
        G_t = torch.as_tensor(G).to(device=device)

        # Get the action probabilities
        action_probabilities = self.forward(state)

        ## Compute the losses
        # Entropy loss
        entropy_loss = -Categorical(probs=action_probabilities).entropy()

        # REINFORCE loss
        policy_loss = -G_t * torch.log(action_probabilities[action])

        # Linear combination of both
        self.loss = policy_loss + Config.ENTROPY_FACTOR * entropy_loss

        # Logging
        self.writer.add_scalar("Action/proba_left", action_probabilities[0], self.index)
        self.writer.add_scalar("Loss/entropy", entropy_loss, self.index)
        self.writer.add_scalar("Loss/policy", policy_loss, self.index)
        self.writer.add_scalar("Loss/loss", self.loss, self.index)

        # Zero the gradient
        self.optimizer.zero_grad()

        # Backprop
        self.loss.backward()

        # Update the parameters
        self.optimizer.step()

    def save(self, name: str = "model"):
        """
        Save the current model

        Args:
            name (str, optional): [Name of the model]. Defaults to "model".
        """
        torch.save(self.policy_network, f"{Config.MODEL_PATH}/{name}.pth")

    def load(self, name: str = "model"):
        """
        Load the designated model

        Args:
            name (str, optional): The model to be loaded (it should be in the "models" folder). Defaults to "model".
        """
        self.policy_network = torch.load(f"{Config.MODEL_PATH}/{name}.pth")


def test_NN(env):
    print(f"Using {device} device")
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.n
    model = PolicyNetwork(obs_shape, action_shape).to(device)

    observation = env.reset()
    S = torch.tensor(observation, device=device)
    R = torch.tensor([1.0], device=device)
    A = torch.tensor([1], device=device)
    pred_probab = model(S)
    model.update_policy(S, A, R)
    print(f"Predicted probas: {pred_probab}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    test_NN(env)

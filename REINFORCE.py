# For type hinting only
from textwrap import wrap
from typing import List
import gym

# Base class for Agent
from agent import Agent

from utils import wrap_env

# The network we create and the device to run it on
from network import PolicyNetwork, device

# Numpy
import numpy as np

# For logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Load config
from config import Config

# Initialize Tensorboard
writer = SummaryWriter()


class REINFORCE(Agent):
    def __init__(self, env: gym.Env) -> None:
        super(Agent, self).__init__()

        # Underlying Gym env
        self.env = env

        # Fetch the action and state space from the underlying Gym environment
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.n

        # Initialize the policy network with the right shape
        self.policy_network = PolicyNetwork(self.obs_shape, self.action_shape).to(
            device
        )

        # Define batch size from Config
        self.batch_size = Config.BATCH_SIZE

        # For logging purpose
        self.policy_network.writer = writer
        self.global_idx_episode = 0
        self.best_episode_reward = 0

    def collect_rollout(self, env: gym.Env, nb_episodes: int) -> list:
        """
        Collect trajectories (states, actions, rewards) by interacting with the environment given the curent policy.

        Args:
            env (gym.Env): Environment to interact with
            nb_episodes (int): Number of environment episodes to play

        Returns:
            list: List of trajectories for each episode. An episode is a list of dictionnary for each timestep stating the state action and reward
        """

        # Init both lists
        rollout = []
        rewards = []

        # Collect for nb_episodes
        for idx_episode in range(nb_episodes):

            # Init episode and reward sum
            episode = []
            reward_sum = 0
            obs = env.reset()
            done = False
            t = 0

            # Generate episode
            while not done:
                # Select action using our current policy
                action = self.select_action(obs)

                # Step the environment given the selected action
                next_obs, reward, done, _ = env.step(action)

                # Collect the state, action and reward. Update the reward sum.
                reward_sum += reward
                step = {"timestep": t, "obs": obs, "action": action, "reward": reward}
                obs = next_obs

                # Add the step to the episode
                episode.append(step)

                # Next step
                t += 1

            # Add the episode to the episode and add the cumulative reward to the rewards
            rollout.append(episode)
            rewards.append(reward_sum)

            # Save best model if new best
            if reward_sum > self.best_episode_reward:
                self.best_episode_reward = reward_sum
                self.save("best")

            # Next episode
            self.global_idx_episode += 1

            # Logging on Tensorboard
            writer.add_scalar(
                "Reward/collection", np.mean(rewards), self.global_idx_episode
            )

        return rollout

    def train(self, env: gym.Env, nb_episodes_per_epoch: int, nb_epoch: int) -> None:
        """
        Train the agent : Collect rollouts and update the policy network.


        Args:
            env (gym.Env): The Gym environment to train on
            nb_episodes_per_epoch (int): Number of episodes per epoch. How much episode to run before updating the policy
            nb_epoch (int): Number of epochs to train on.
        """
        # For logging purposes
        self.global_idx_episode = 0

        # Iterate over epochs
        for epoch in tqdm(range(nb_epoch)):

            # Collect rollout for this epoch
            rollout = self.collect_rollout(env, nb_episodes_per_epoch)

            # Update the policy using the episodes in the rollout
            for episode in rollout:

                # If the episode is shorter than the batch size, select the min to prevent bug.0
                batch_size = min(len(episode), self.batch_size)

                # Compute the returns of the episode
                episode = self.compute_return(episode, gamma=Config.GAMMA)

                # Initialize batch
                batch = {"states": [], "returns": [], "actions": []}

                # Iterate over the episode, t is the timestep number and timestep is the actual dictionnary representing the timestep
                for t, timestep in enumerate(episode):

                    # Collect the state, return and action in the batch
                    batch["states"].append(timestep["obs"])
                    batch["returns"].append(timestep["return"])
                    batch["actions"].append(timestep["action"])

                    # Every batch_size timesteps, update the policy using the normalized returns
                    if t > 0 and t % batch_size == 0 and batch_size > 1:

                        # Compute mean and standard deviation of the returns of the batch for normalization
                        return_mean = np.mean(batch["returns"])
                        return_std = np.std(batch["returns"])

                        # Normalize the returns
                        for i, return_ in enumerate(batch["returns"]):
                            batch["returns"][i] = (return_ - return_mean) / return_std
                        for state, return_, action in zip(
                            batch["states"], batch["returns"], batch["actions"]
                        ):
                            # Update policy for every timestep in the batch
                            self.policy_network.update_policy(state, action, return_)

                        # Next batch
                        batch = {"states": [], "returns": [], "actions": []}

                    # Safe case for batch size of one (where no normalization is done)
                    elif self.batch_size == 1:
                        self.policy_network.update_policy(
                            timestep["obs"], timestep["action"], timestep["return"]
                        )

    def select_action(self, observation: np.array, testing: bool = False) -> int:
        """
        Select the action based on the current policy and the observation

        Args:
            observation (np.array): State representation
            testing (bool): Wether to be in test mode or not.

        Returns:
            int: The selected action
        """
        # If test mode, switch to eval mode in PyTorch
        if testing:
            self.policy_network.eval()
        # Infer the action probabilities from the network
        action_probabilities = self.policy_network.get_action_probabilities(observation)

        # Randomly select the action given the action action probabilities
        action = np.random.choice(
            list(range(self.action_shape)), p=action_probabilities
        )
        return int(action)

    def compute_return(self, episode: list, gamma: float = 0.99) -> list:
        """
        Compute the discounted return for each step of the episode

        Args:
            episode (list): The episode to compute returns on. A list of dictionnaries for each timestep containing state action and reward.
            gamma (float, optional): The discount factor. Defaults to 0.99.

        Raises:
            ValueError: If gamma is not valid (not between 0 and 1)

        Returns:
            list: The updated episode list of dictionnaries with a "return" key added for each timestep.
        """
        if not (0 <= gamma <= 1):
            raise ValueError(f"Gamma not between 0 and 1, gamma = {gamma}")

        # Init return
        G = 0

        # Iterate over the episode starting from the end
        for i, step in enumerate(reversed(episode)):
            # Compute return using the return formula
            G = step["reward"] + (gamma ** i) * G

            # Add the return to the timestep dictionnary
            step["return"] = G

            # Change the corresponding timestep of the episode by the new step
            episode[len(episode) - i - 1] = step

        return episode

    def test(self, env: gym.Env, nb_episodes: int, render: bool = False) -> None:
        """
        Test the current policy to evalute its performance

        Args:
            env (gym.Env): The Gym environment to test it on
            nb_episodes (int): Number of test episodes
            render (bool, optional): Wether or not to render the visuals of the episodes while testing. Defaults to False.
        """

        env = wrap_env(env)

        # Iterate over the episodes
        for episode in range(nb_episodes):
            # Init episode
            done = False
            obs = env.reset()
            rewards_sum = 0

            # Generate episode
            while not done:
                # Select the action using the current policy
                action = self.select_action(obs)

                # Step the environment accordingly
                next_obs, reward, done, _ = env.step(action)

                # Log reward for performance tracking
                rewards_sum += reward

                # render the environment
                if render:
                    env.render()

                # Next step
                obs = next_obs

            # Logging
            writer.add_scalar("Reward/test", rewards_sum, episode)
            print(f"test number {episode} : {rewards_sum}")

    def save(self, name: str = "model"):
        """
        Wrapper method for saving the network weights.

        Args:
            name (str, optional): Name of the save model file. Defaults to "model".
        """
        self.policy_network.save(name)

    def load(self, name: str):
        """
        Wrapper method for loading the network weights.

        Args:
            name (str, optional): Name of the save model file.
        """
        self.policy_network.load(name)

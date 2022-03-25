# For type hinting only
from typing import List
import gym

# Base class for Agent
from agent import Agent

# The network we create and the device to run it on
from network import ActorCritic, device

# Numpy
import numpy as np

# For logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Load config
from config import Config

from datetime import datetime
from datetime import date

now = datetime.now()

current_time = now.strftime("%H:%M")

today = date.today().strftime("%d-%m-%Y")

# Initialize Tensorboard
writer = SummaryWriter(log_dir=f"{Config.TENSORBOARD_PATH}/{today}/{current_time}")


class A2C(Agent):
    def __init__(self, env: gym.Env) -> None:
        super(Agent, self).__init__()

        # Underlying Gym env
        self.env = env

        # Fetch the action and state space from the underlying Gym environment
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.n

        # Initialize the policy network with the right shape
        self.network = ActorCritic(self.obs_shape, self.action_shape).to(device)

        # Define batch size from Config
        self.batch_size = Config.BATCH_SIZE

        # For logging purpose
        self.network.writer = writer
        self.global_idx_episode = 0
        self.best_episode_reward = 0

    def select_action(self, observation: np.array, testing: bool = False) -> int:
        """
        Select the action based on the current policy and the observation

        Args:
            observation (np.array): State representation
            testing (bool): Wether to be in test mode or not.

        Returns:
            int: The selected action
        """
        return self.network.select_action(observation)

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
        t = 0
        for epoch in tqdm(range(nb_epoch)):
            batch = []
            for episode in range(nb_episodes_per_epoch):
                reward_sum = 0
                done, obs = False, env.reset()
                while not done:
                    action = self.select_action(obs)
                    next_obs, reward, done, _ = env.step(action)
                    self.network.update_policy(obs, action, reward, next_obs, done)
                    reward_sum += reward
                    obs = next_obs
                    t += 1
                self.global_idx_episode += 1
                if reward_sum > self.best_episode_reward:
                    self.best_episode_reward = reward_sum
                    self.save("best")
                self.network.writer.add_scalar(
                    "Reward/Episode_sum_of_rewards", reward_sum, self.global_idx_episode
                )

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
        self.network.save(name)

    def load(self, name: str):
        """
        Wrapper method for loading the network weights.

        Args:
            name (str, optional): Name of the save model file.
        """
        self.network.load(name)

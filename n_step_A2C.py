# For type hinting only
from typing import List
import gym

# Base class for Agent
from agent import Agent

# The network we create and the device to run it on
from network import ActorCritic, Critic, device

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


class Memory:
    def __init__(self, n_steps):
        self.steps = {"states": [], "actions": [], "rewards": [], "dones": []}
        self.n_steps = n_steps

    def add(self, state, action, reward, done):
        self.steps["states"].append(state)
        self.steps["actions"].append(action)
        self.steps["rewards"].append(reward)
        self.steps["dones"].append(done)

    def clear(self):
        self.steps = {"states": [], "actions": [], "rewards": [], "dones": []}

    def remove_first_step(self):
        self.steps = {key: values[1:] for key, values in self.steps.items()}

    def compute_return(self):
        n_step_return = 0
        for i, reward in enumerate(reversed(self.steps["rewards"])):
            n_step_return += (
                reward
                + (1.0 - self.steps["dones"][i]) * Config.GAMMA ** i * n_step_return
            )
        return (
            n_step_return,
            self.steps["states"][0],
            self.steps["actions"][0],
            self.steps["dones"][0],
        )

    def get_step(self, i):
        return {key: values[i] for key, values in self.steps.items()}

    # def _zip(self):
    #     return zip(
    #         self.states[: self.n_steps],
    #         self.actions[: self.n_steps],
    #         self.rewards[: self.n_steps],
    #         self.dones[: self.n_steps],
    #         self.n_step_returns,
    #     )

    # def reversed(self):
    #     for data in list(self._zip())[::-1]:
    #         yield data

    def __len__(self):
        return len(self.steps["rewards"])


class A2C(Agent):
    def __init__(self, env: gym.Env) -> None:
        super(Agent, self).__init__()

        # Underlying Gym env
        self.env = env
        self.memory = Memory(n_steps=Config.N_STEPS)
        # Fetch the action and state space from the underlying Gym environment
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.n

        # Initialize the policy network with the right shape
        self.network = ActorCritic(self.obs_shape, self.action_shape).to(device)

        # Define batch size from Config
        self.batch_size = Config.BATCH_SIZE

        # For logging purpose
        self.network.writer = writer
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

    def train(self, env: gym.Env, nb_timestep: int) -> None:
        """
        Train the agent : Collect rollouts and update the policy network.


        Args:
            env (gym.Env): The Gym environment to train on
            nb_episodes_per_epoch (int): Number of episodes per epoch. How much episode to run before updating the policy
            nb_epoch (int): Number of epochs to train on.
        """

        episode = 1
        t = 1
        # Iterate over epochs
        while t <= nb_timestep:

            # Init reward_sum and variables for the episode
            reward_sum = 0
            done, obs = False, env.reset()

            # Loop through the episode
            while not done:

                # Select the action using the actor network
                action = self.select_action(obs)

                # Step the environment
                next_obs, reward, done, _ = env.step(action)

                # Add the experience collected to the memory for the n-step processing
                self.memory.add(obs, action, reward, done)

                # When we have collected n steps we can start learning
                if t >= Config.N_STEPS:
                    # Compute the n-steps return to be used as target and fetch the relevant information from the memory
                    (
                        n_step_return,
                        old_obs,
                        old_action,
                        old_done,
                    ) = self.memory.compute_return()
                    # Run the n-step A2C update
                    self.network.update_policy(
                        old_obs, old_action, n_step_return, next_obs, done
                    )
                    # Clear the used experience from the memory
                    self.memory.remove_first_step()

                # Update timesteps counter, reward sum and move on to the next observation
                t += 1
                reward_sum += reward
                obs = next_obs

            # Clear memory to start a new episode
            self.memory.clear()

            # Track best model and save it
            if reward_sum > self.best_episode_reward:
                self.best_episode_reward = reward_sum
                self.save("best")

            # Log performances in Tensorboard
            self.network.writer.add_scalar(
                "Reward/Episode_sum_of_rewards", reward_sum, episode
            )
            # Next episode
            episode += 1

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

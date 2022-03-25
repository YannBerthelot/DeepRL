# For type hinting only
from typing import List
import gym

# Base class for Agent
from agent import Agent

# The network we create and the device to run it on
from network import ActorCritic

# Numpy
import numpy as np

# For logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datetime import datetime
from datetime import date

now = datetime.now()


class Memory:
    def __init__(self, n_steps, config):
        self.steps = {"states": [], "actions": [], "rewards": [], "dones": []}
        self.n_steps = n_steps
        self.config = config

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
                + (1.0 - self.steps["dones"][i])
                * self.config["GAMMA"] ** i
                * n_step_return
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
    def __init__(self, env: gym.Env, config: dict, comment: str = "") -> None:
        super(Agent, self).__init__()

        # current_time = now.strftime("%H:%M")
        today = date.today().strftime("%d-%m-%Y")
        LOG_DIR = (
            f'{config["TENSORBOARD_PATH"]}/{config["ENVIRONMENT"]}/{today}/{comment}'
        )
        # Initialize Tensorboard
        writer = SummaryWriter(log_dir=LOG_DIR)

        # Underlying Gym env
        self.env = env
        self.__name__ = "n-steps A2C"
        self.memory = Memory(n_steps=config["N_STEPS"], config=config)
        # Fetch the action and state space from the underlying Gym environment
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.n
        self.config = config
        # Initialize the policy network with the right shape
        self.network = ActorCritic(self.obs_shape, self.action_shape, config=config)
        self.network.to(self.network.device)

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
        # Early stopping
        constant_reward_counter = 0
        old_reward_sum = 0

        # Init training
        episode = 1
        t = 1
        t_old = 0
        # Iterate over epochs
        pbar = tqdm(total=nb_timestep, initial=1)
        while t <= nb_timestep:
            pbar.update(t - t_old)
            t_old = t

            # Init reward_sum and variables for the episode
            rewards = []
            done, obs = False, env.reset()

            # Loop through the episode
            while not done:

                # Select the action using the actor network
                action = self.select_action(obs)

                # Step the environment
                next_obs, reward, done, _ = env.step(action)
                rewards.append(reward)

                # Add the experience collected to the memory for the n-step processing
                self.memory.add(obs, action, reward, done)

                # When we have collected n steps we can start learning
                if t >= self.config["N_STEPS"]:
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
                obs = next_obs

            # Clear memory to start a new episode
            self.memory.clear()
            reward_sum = np.sum(rewards)

            # Track best model and save it
            if reward_sum > self.best_episode_reward:
                self.best_episode_reward = reward_sum
                self.save("best")
            elif reward_sum == old_reward_sum:
                constant_reward_counter += 1
                if constant_reward_counter > 10:
                    break
            old_reward_sum = reward_sum
            # Log performances in Tensorboard
            self.network.writer.add_scalar(
                "Reward/Episode_sum_of_rewards", reward_sum, episode
            )
            self.network.writer.add_histogram(
                "Reward distribution",
                np.array(
                    [
                        np.mean(rewards),
                        np.std(rewards),
                        -np.std(rewards),
                        max(rewards),
                        min(rewards),
                    ]
                ),
                episode,
            )
            # Next episode
            episode += 1
        # self.network.writer.add_hparams(
        #     config,
        #     {
        #         "train mean reward": np.mean(rewards),
        #         "train std reward": np.std(rewards),
        #         "train max reward": max(rewards),
        #         "train test reward": min(rewards),
        #     },
        #     run_name="",
        # )
        pbar.close()

    def test(self, env: gym.Env, nb_episodes: int, render: bool = False) -> None:
        """
        Test the current policy to evalute its performance

        Args:
            env (gym.Env): The Gym environment to test it on
            nb_episodes (int): Number of test episodes
            render (bool, optional): Wether or not to render the visuals of the episodes while testing. Defaults to False.
        """
        episode_rewards = []
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
            self.network.writer.add_scalar("Reward/test", rewards_sum, episode)
            print(f"test number {episode} : {rewards_sum}")
            episode_rewards.append(rewards_sum)
        self.network.writer.add_hparams(
            self.config,
            {
                "test mean reward": np.mean(episode_rewards),
                "test std reward": np.std(episode_rewards),
                "test max reward": max(episode_rewards),
                "min test reward": min(episode_rewards),
            },
            run_name=".",
        )

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

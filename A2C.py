# For type hinting only
import os
import pickle
import gym
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import SimpleMinMaxScaler, SimpleStandardizer
import wandb

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


class RolloutBuffer:
    def __init__(self, n_steps, config) -> None:
        self.steps = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "hiddens": [],
        }
        self.n_steps = n_steps
        self.config = config
        pass

    def add(self, state, action, reward, done, hidden=None):
        self.steps["states"].append(state)
        self.steps["actions"].append(action)
        self.steps["rewards"].append(reward)
        self.steps["dones"].append(done)
        self.steps["hiddens"].append(hidden)

    def clear(self):
        self.steps = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "hiddens": [],
        }

    def get_steps(self):
        return self.steps


class Memory:
    def __init__(self, n_steps, config):
        self.steps = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "hiddens": [],
        }
        self.n_steps = n_steps
        self.config = config

    def add(self, state, action, reward, done, hidden=None):
        self.steps["states"].append(state)
        self.steps["actions"].append(action)
        self.steps["rewards"].append(reward)
        self.steps["dones"].append(done)
        self.steps["hiddens"].append(hidden)

    def clear(self):
        self.steps = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "hiddens": [],
        }

    def remove_first_step(self):
        self.steps = {key: values[1:] for key, values in self.steps.items()}

    def compute_return(self):
        n_step_return = 0
        for i, reward in enumerate(reversed(self.steps["rewards"])):
            n_step_return = (
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
            self.steps["hiddens"][0],
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
    def __init__(self, env: gym.Env, config: dict, comment: str = "", run=None) -> None:
        super(Agent, self).__init__()

        # current_time = now.strftime("%H:%M")
        today = date.today().strftime("%d-%m-%Y")
        LOG_DIR = (
            f'{config["TENSORBOARD_PATH"]}/{config["ENVIRONMENT"]}/{today}/{comment}'
        )
        os.makedirs(config["MODEL_PATH"], exist_ok=True)
        # Initialize Tensorboard
        writer = SummaryWriter(log_dir=LOG_DIR)
        self.comment = comment
        # Underlying Gym env
        self.env = env
        self.__name__ = "n-steps A2C"
        self.memory = Memory(n_steps=config["N_STEPS"], config=config)
        # Fetch the action and state space from the underlying Gym environment
        self.obs_shape = env.observation_space.shape
        if config["RECURRENT"] and self.config["ADD_ACTION"]:
            self.obs_shape = (self.obs_shape[0] + 1,)
        if config["CONTINUOUS"]:
            self.action_shape = env.action_space.shape[0]
        else:
            self.action_shape = env.action_space.n
        self.config = config
        if self.config["SCALING"]:

            if self.config["SCALING_METHOD"] == "standardize":
                self.obs_scaler = SimpleStandardizer(clip=True)
                self.reward_scaler = SimpleStandardizer(shift_mean=False, clip=False)
                self.target_scaler = SimpleStandardizer(shift_mean=False, clip=False)
            elif self.config["SCALING_METHOD"] == "normalize":
                self.obs_scaler = MinMaxScaler(feature_range=(-1, 1))
                self.reward_scaler = SimpleMinMaxScaler(
                    maxs=[100], mins=[-100], feature_range=(-1, 1)
                )
        else:
            self.config["LEARNING_START"] = 0
            self.obs_scaler = None
            self.reward_scaler = None
            self.target_scaler = None

        # Initialize the policy network with the right shape
        self.network = ActorCritic(
            self.obs_shape,
            self.action_shape,
            config=config,
            scaler=self.target_scaler,
            env=env,
        )
        self.network.to(self.network.device)
        self.run = run

        # For logging purpose
        self.network.writer = writer
        self.best_episode_reward = -np.inf

    def select_action(
        self, observation: np.array, hidden: np.array, testing: bool = False
    ) -> int:
        """
        Select the action based on the current policy and the observation

        Args:
            observation (np.array): State representation
            testing (bool): Wether to be in test mode or not.

        Returns:
            int: The selected action
        """
        action, next_hidden = self.network.select_action(observation, hidden)
        return action, next_hidden

    def train(self, env: gym.Env, nb_timestep: int) -> None:
        """
        Train the agent : Collect rollouts and update the policy network.


        Args:
            env (gym.Env): The Gym environment to train on
            nb_episodes_per_epoch (int): Number of episodes per epoch. How much episode to run before updating the policy
            nb_epoch (int): Number of epochs to train on.
        """
        # Early stopping
        self.constant_reward_counter, self.old_reward_sum = 0, 0

        # Init training
        self.episode, self.t, t_old, self.constant_reward_counter = 1, 1, 0, 0

        # Iterate over epochs
        pbar = tqdm(total=nb_timestep, initial=1)

        while self.t <= nb_timestep:
            # tqdm stuff
            pbar.update(self.t - t_old)
            t_old, t_episode = self.t, 0

            done, obs, rewards, hidden = False, env.reset(), []
            hidden = self.network.get_initial_states()

            if self.config["SCALING"]:
                self.obs_scaler.partial_fit(obs)

            while not done:
                action, next_hidden = self.select_action(obs, hidden)
                next_obs, reward, done, _ = env.step(action)
                rewards.append(reward)
                reward, next_obs = self.scaling(reward, next_obs)

                # Add the experience collected to the memory for the n-step processing
                if self.t >= self.config["LEARNING_START"]:
                    self.memory.add(obs, action, reward, done, hidden)

                    # When we have collected n steps we can start learning
                    if t_episode >= self.config["N_STEPS"]:
                        # Compute the n-steps return to be used as target and fetch the relevant information from the memory
                        (
                            n_step_return,
                            old_obs,
                            old_action,
                            old_done,
                            old_hidden,
                        ) = self.memory.compute_return()
                        if not done:
                            self.network.update_policy(
                                old_obs,
                                old_action,
                                n_step_return,
                                next_obs,
                                old_hidden,
                                hidden,
                                done,
                            )
                        self.memory.remove_first_step()
                self.t, t_episode = self.t + 1, t_episode + 1
                obs, hidden = next_obs, next_hidden

            self.memory.clear()
            reward_sum = np.sum(rewards)

            # Track best model and save it
            self.save_if_best(reward_sum)
            if self.early_stopping(reward_sum):
                break

            self.old_reward_sum, self.episode = reward_sum, self.episode + 1
            self.episode_logging(rewards, reward_sum)

        pbar.close()
        self.train_logging()

    def test(
        self, env: gym.Env, nb_episodes: int, render: bool = False, scaler_file=None
    ) -> None:
        """
        Test the current policy to evalute its performance

        Args:
            env (gym.Env): The Gym environment to test it on
            nb_episodes (int): Number of test episodes
            render (bool, optional): Wether or not to render the visuals of the episodes while testing. Defaults to False.
        """
        if scaler_file is not None:
            with open(scaler_file, "rb") as input_file:
                scaler = pickle.load(input_file)
            self.obs_scaler = scaler
        episode_rewards = []
        best_test_episode_reward = 0
        # Iterate over the episodes
        for episode in tqdm(range(nb_episodes)):
            if self.config["RECURRENT"]:
                hidden = self.network.get_initial_states()
            else:
                hidden = None
            # Init episode
            done = False
            obs = env.reset()
            rewards_sum = 0

            # Generate episode
            while not done:
                # Select the action using the current policy
                if self.config["SCALING"]:
                    obs = self.obs_scaler.transform(obs)
                action, next_hidden = self.select_action(obs, hidden)

                # Step the environment accordingly
                next_obs, reward, done, _ = env.step(action)

                # Log reward for performance tracking
                rewards_sum += reward

                # render the environment
                if render:
                    env.render()

                # Next step
                obs = next_obs
                hidden = next_hidden

            if rewards_sum > best_test_episode_reward:
                best_test_episode_reward = rewards_sum
                if self.config["logging"] == "wandb":
                    wandb.run.summary["Test/best reward sum"] = rewards_sum
            # Logging
            if self.config["logging"] == "wandb":
                wandb.log(
                    {"Test/reward": rewards_sum, "Test/episode": episode}, commit=True
                )
            elif self.config["logging"] == "tensorboard":
                self.network.writer.add_scalar("Reward/test", rewards_sum, episode)
            # print(f"test number {episode} : {rewards_sum}")
            episode_rewards.append(rewards_sum)
        if self.config["logging"] == "tensorboard":
            self.network.writer.add_hparams(
                self.config,
                {
                    "test mean reward": np.mean(episode_rewards),
                    "test std reward": np.std(episode_rewards),
                    "test max reward": max(episode_rewards),
                    "min test reward": min(episode_rewards),
                },
                run_name="test",
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

    def save_if_best(self, reward_sum):
        if reward_sum >= self.best_episode_reward:
            self.best_episode_reward = reward_sum
            if self.config["logging"] == "wandb":
                wandb.run.summary["Train/best reward sum"] = reward_sum
                artifact = wandb.Artifact(f"{self.comment}_best", type="model")

            self.save(f"{self.comment}_best")

    def early_stopping(self, reward_sum):
        if reward_sum == self.old_reward_sum:
            self.constant_reward_counter += 1
            if self.constant_reward_counter > self.config["EARLY_STOPPING_STEPS"]:
                print(
                    f'Early stopping due to constant reward for {self.config["EARLY_STOPPING_STEPS"]} steps'
                )
                return True
        else:
            self.constant_reward_counter = 0
        return False

    def episode_logging(self, rewards, reward_sum):
        if self.config["logging"] == "wandb":
            wandb.log(
                {
                    "Train/Episode_sum_of_rewards": reward_sum,
                    "Train/Episode": self.episode,
                },
                step=self.t,
                commit=True,
            )
        elif self.config["logging"] == "tensorboard":
            self.network.writer.add_scalar(
                "Reward/Episode_sum_of_rewards", reward_sum, self.episode
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
                self.episode,
            )

    def train_logging(self):
        if self.config["logging"] == "wandb":
            # Add a file to the artifact's contents
            artifact.add_file(f'{self.config["MODEL_PATH"]}/{self.comment}_best.pth')

            # Save the artifact version to W&B and mark it as the output of this run
            self.run.log_artifact(artifact)

            artifact = wandb.Artifact(f"{self.comment}_obs_scaler", type="scaler")

            # Add a file to the artifact's contents
            pickle.dump(
                self.obs_scaler,
                open(f"data/{self.comment}_obs_scaler.pkl", "wb"),
            )
            artifact.add_file(f"data/{self.comment}_obs_scaler.pkl")

            # Save the artifact version to W&B and mark it as the output of this run
            self.run.log_artifact(artifact)

    def scaling(self, reward, next_obs):
        # Scaling
        if self.config["SCALING"]:
            if self.t >= self.config["LEARNING_START"]:
                # self.obs_scaler.partial_fit(next_obs)
                # self.reward_scaler.partial_fit(np.array([reward]))
                reward = self.reward_scaler.transform(np.array([reward]))[0]
                next_obs = self.obs_scaler.transform(next_obs)
            else:
                self.obs_scaler.partial_fit(next_obs)
                self.reward_scaler.partial_fit(np.array([reward]))
        return reward, next_obs

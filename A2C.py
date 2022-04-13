import os
import pickle
import gym
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from normalize import SimpleMinMaxScaler, SimpleStandardizer, RunningMeanStd
import wandb

# Base class for Agent
from agent import Agent
from buffer import Memory, RolloutBuffer

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


class A2C(Agent):
    def __init__(self, env: gym.Env, config: dict, comment: str = "", run=None) -> None:
        super(Agent, self).__init__()
        self.config = config
        self.env = env
        self.comment = comment

        self.memory = Memory(n_steps=config["N_STEPS"], config=config)
        self.rollout = RolloutBuffer(buffer_size=config["N_STEPS"])
        self.obs_shape = env.observation_space.shape
        self.action_shape = (
            env.action_space.shape[0] if config["CONTINUOUS"] else env.action_space.n
        )

        self.obs_scaler, self.reward_scaler, self.target_scaler = self.get_scalers()
        self.obs_scaler = RunningMeanStd(shape=env.observation_space.shape)

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
        LOG_DIR = self.create_dirs()
        writer = SummaryWriter(log_dir=LOG_DIR)
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
            t_old, t_episode = self.t, 1

            done, obs, rewards = False, env.reset(), []
            hidden = self.network.get_initial_states()
            while not done:
                if self.t > self.config["LEARNING_START"]:
                    action, next_hidden = self.select_action(obs, hidden)

                    action = action.detach().data.numpy()
                else:
                    action, next_hidden = self.env.action_space.sample(), hidden
                next_obs, reward, done, _ = env.step(action)
                rewards.append(reward)
                next_obs, reward = self.scaling(next_obs, reward)
                if self.t >= self.config["LEARNING_START"]:
                    self.rollout.add(
                        obs, next_obs, action, reward, done, hidden, next_hidden
                    )
                if (
                    (t_episode > 1)
                    and (t_episode % self.config["BUFFER_SIZE"] == 0)
                    and (self.t > self.config["LEARNING_START"])
                ):
                    self.network.update_policy(self.rollout)
                    self.rollout.reset()
                self.t, t_episode = self.t + 1, t_episode + 1
                obs, hidden = next_obs, next_hidden
            reward_sum = np.sum(rewards)
            self.rollout.reset()
            # Track best model and save it
            artifact = self.save_if_best(reward_sum)
            if self.early_stopping(reward_sum):
                break

            self.old_reward_sum, self.episode = reward_sum, self.episode + 1
            self.episode_logging(rewards, reward_sum)

        pbar.close()
        self.train_logging(artifact)

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
                next_obs, reward, done, _ = env.step(action.detach().data.numpy())

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
        artifact = None
        if reward_sum >= self.best_episode_reward:
            self.best_episode_reward = reward_sum
            if self.config["logging"] == "wandb":
                wandb.run.summary["Train/best reward sum"] = reward_sum
                artifact = wandb.Artifact(f"{self.comment}_best", type="model")

            self.save(f"{self.comment}_best")
        return artifact

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

    def train_logging(self, artifact):
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

    def scaling(self, obs, reward):
        # Scaling
        if self.config["SCALING"]:
            if self.t >= self.config["LEARNING_START"]:
                self.obs_scaler.partial_fit(obs)
                self.reward_scaler.partial_fit(np.array([reward]))
                reward = self.reward_scaler.transform(np.array([reward]))[0]
                obs = self.obs_scaler.transform(obs)
            else:
                self.obs_scaler.partial_fit(obs)
                self.reward_scaler.partial_fit(np.array([reward]))
        return obs, reward

    def create_dirs(self):
        today = date.today().strftime("%d-%m-%Y")
        os.makedirs(self.config["MODEL_PATH"], exist_ok=True)
        return f'{self.config["TENSORBOARD_PATH"]}/{self.config["ENVIRONMENT"]}/{today}/{self.comment}'

    def get_scalers(self):
        if self.config["SCALING"]:
            if self.config["SCALING_METHOD"] == "standardize":
                obs_scaler = SimpleStandardizer(clip=True)
                reward_scaler = SimpleStandardizer(shift_mean=False, clip=False)
                target_scaler = SimpleStandardizer(shift_mean=False, clip=False)
            elif self.config["SCALING_METHOD"] == "normalize":
                obs_scaler = MinMaxScaler(feature_range=(-1, 1))
                reward_scaler = SimpleMinMaxScaler(
                    maxs=[100], mins=[-100], feature_range=(-1, 1)
                )
        else:
            self.config["LEARNING_START"] = 0
            obs_scaler = None
            reward_scaler = None
            target_scaler = None
        return obs_scaler, reward_scaler, target_scaler

    # def scaling(self, obs, reward):
    #     if self.config["SCALING"]:
    #         self.obs_scaler.partial_fit(obs)
    #         self.reward_scaler.partial_fit(reward)
    #         if self.config["LEARNING_START"]:
    #             obs = self.obs_scaler.transform(obs)
    #             reward = self.reward_scaler.transform(reward)
    #     return obs, reward

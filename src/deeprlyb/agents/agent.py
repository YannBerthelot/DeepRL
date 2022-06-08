import os
import pickle
from abc import abstractmethod
from datetime import date
import gym
import wandb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from deeprlyb.utils.normalize import (
    SimpleMinMaxScaler,
    SimpleStandardizer,
    RunningMeanStd,
)


class Agent:
    """
    Base class for agents
    """

    def __init__(self, env: gym.Env, config: dict, comment: str = "", run=None) -> None:
        self._env = env
        self._config = config
        self.comment = comment
        self.run = run

        # For logging purpose
        self.log_dir = self.create_dirs()
        self.best_episode_reward, self.episode = -np.inf, 1

    @property
    def config(self):
        return self._config

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, new_env: gym.Env):
        if (new_env.observation_space == self._env.observation_space) & (
            new_env.action_space == self._env.action_space
        ):
            self._env = new_env
        else:
            raise ValueError("Environment spaces don't match. Check new environment.")

    @property
    def obs_shape(self):
        return self.env.observation_space.shape

    @property
    def action_shape(self):
        return (
            self.env.action_space.shape[0]
            if self.config["GLOBAL"].getboolean("continuous")
            else self.env.action_space.n
        )

    @abstractmethod
    def select_action(self, observation):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self, env: gym.Env, nb_episodes: int, render: bool = False) -> None:
        raise NotImplementedError

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
            if self.config["GLOBAL"]["logging"] == "wandb":
                wandb.run.summary["Train/best reward sum"] = reward_sum
                artifact = wandb.Artifact(f"{self.comment}_best", type="model")

            self.save(f"{self.comment}_best")
        return artifact

    def early_stopping(self, reward_sum):
        if reward_sum == self.old_reward_sum:
            self.constant_reward_counter += 1
            if self.constant_reward_counter > self.config["GLOBAL"].getint(
                "early_stopping_steps"
            ):
                print(
                    f'Early stopping due to constant reward for {self.config["GLOBAL"].getint("early_stopping_steps")} steps'
                )
                return True
        else:
            self.constant_reward_counter = 0
        return False

    def episode_logging(self, rewards: list, reward_sum: float, actions_taken: dict):
        n_actions = sum(actions_taken.values())
        action_frequencies = {
            action: n / n_actions for action, n in actions_taken.items()
        }
        if self.config["GLOBAL"]["logging"] == "wandb":
            for action, frequency in action_frequencies.items():
                wandb.log(
                    {
                        f"Actions/{action}": frequency,
                    },
                    step=self.t,
                    commit=False,
                )
            wandb.log(
                {
                    "Train/Episode_sum_of_rewards": reward_sum,
                    "Train/Episode": self.episode,
                },
                step=self.t,
                commit=True,
            )
        elif self.config["GLOBAL"]["logging"] == "tensorboard":
            self.network.writer.add_scalar(
                "Reward/Episode_sum_of_rewards", reward_sum, self.episode
            )

    def scaling(self, obs, reward, fit=True, transform=True):
        # Scaling
        if self.config["GLOBAL"].getboolean("scaling"):
            if fit:
                self.obs_scaler.partial_fit(obs)
                self.reward_scaler.partial_fit(np.array([reward]))
            if transform:
                reward = self.reward_scaler.transform(np.array([reward]))[0]
                obs = self.obs_scaler.transform(obs)
        return obs, reward

    def create_dirs(self):
        today = date.today().strftime("%d-%m-%Y")
        os.makedirs(self.config["PATHS"]["model_path"], exist_ok=True)
        return f'{self.config["PATHS"]["tensorboard_path"]}/{self.config["GLOBAL"]["environment"]}/{today}/{self.comment}'

    def get_scalers(self):
        if self.config["GLOBAL"].getboolean("SCALING"):
            if self.config["GLOBAL"]["scaling_method"] == "standardize":
                obs_scaler = SimpleStandardizer(clip=True)
                reward_scaler = SimpleStandardizer(shift_mean=False, clip=False)
                target_scaler = SimpleStandardizer(shift_mean=False, clip=False)
            elif self.config["GLOBAL"]["scaling_method"] == "normalize":
                obs_scaler = MinMaxScaler(feature_range=(-1, 1))
                reward_scaler = SimpleMinMaxScaler(
                    maxs=[100], mins=[-100], feature_range=(-1, 1)
                )
        else:
            self.config["GLOBAL"]["learning_start"] = "0"
            obs_scaler = None
            reward_scaler = None
            target_scaler = None
        return obs_scaler, reward_scaler, target_scaler

    def train_logging(self, artifact):
        os.makedirs("data", exist_ok=True)
        if self.config["GLOBAL"]["logging"] == "wandb":
            artifact = wandb.Artifact(f"{self.comment}_model", type="model")
            artifact.add_file(
                f'{self.config["PATHS"]["model_path"]}/{self.comment}_best.pth'
            )
            # Save the artifact version to W&B and mark it as the output of this run
            self.run.log_artifact(artifact)

            if self.obs_scaler is not None:
                artifact = wandb.Artifact(f"{self.comment}_obs_scaler", type="scaler")
                pickle.dump(
                    self.obs_scaler,
                    open(f"data/{self.comment}_obs_scaler.pkl", "wb"),
                )
                artifact.add_file(f"data/{self.comment}_obs_scaler.pkl")

                # Save the artifact version to W&B and mark it as the output of this run
                self.run.log_artifact(artifact)
            else:
                print("No scaler to save, skipping")

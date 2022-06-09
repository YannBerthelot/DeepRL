import pickle
import warnings
import gym
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple

# Base class for Agent
from deeprlyb.agents.agent import Agent
from deeprlyb.network.utils import t, compute_KL_divergence, LinearSchedule
from deeprlyb.network.network import ActorCriticRecurrentNetworks, BaseTorchAgent

# Network creator tool
from deeprlyb.utils.normalize import SimpleStandardizer
from deeprlyb.utils.buffer import RolloutBuffer


class A2C(Agent):
    def __init__(self, env: gym.Env, config: dict, comment: str = "", run=None) -> None:
        super(
            A2C,
            self,
        ).__init__(env, config, comment, run)

        self.rollout = RolloutBuffer(
            buffer_size=config["NETWORKS"].getint("buffer_size"),
            gamma=config["AGENT"].getfloat("gamma"),
            n_steps=config["AGENT"].getint("n_steps"),
        )

        self.obs_scaler, self.reward_scaler, self.target_scaler = self.get_scalers()

        # Initialize the policy network with the right shape
        self.network = TorchA2C(self)
        self.network.to(self.network.device)
        self.network.writer = SummaryWriter(log_dir=self.log_dir)

    def select_action(
        self,
        observation: np.array,
        hidden: Dict[int, torch.Tensor],
    ) -> int:
        """
        Select the action based on the current policy and the observation

        Args:
            observation (np.array): State representation
            testing (bool): Wether to be in test mode or not.

        Returns:
            int: The selected action
        """
        return self.network.select_action(observation, hidden)

    def compute_value(self, observation: np.array, hidden: np.array) -> int:
        """
        Select the action based on the current policy and the observation

        Args:
            observation (np.array): State representation
            testing (bool): Wether to be in test mode or not.

        Returns:
            int: The selected action
        """
        return self.network.get_value(observation, hidden)

    def pre_train(self, env: gym.Env, nb_timestep: int) -> None:
        # Init training
        self.t, t_old, self.constant_reward_counter = 1, 0, 0

        # Pre-Training
        if self.config["GLOBAL"].getfloat("learning_start") > 0:
            print("--- Pre-Training ---")
            t_pre_train = 1
            pbar = tqdm(total=nb_timestep, initial=1)
            while t_pre_train <= nb_timestep:
                pbar.update(t_pre_train - t_old)
                t_old = t_pre_train
                done, obs, rewards = False, env.reset(), []
                while not done:
                    action = self.env.action_space.sample()
                    next_obs, reward, done, _ = env.step(action)
                    next_obs, reward = self.scaling(
                        next_obs, reward, fit=True, transform=False
                    )
                    t_pre_train += 1
            pbar.close()
            print(
                f"Obs scaler - Mean : {self.obs_scaler.mean}, std : {self.obs_scaler.std}"
            )
            print(f"Reward scaler - std : {self.reward_scaler.std}")

    def train_TD0(self, env: gym.Env, nb_timestep: int) -> None:
        self.pre_train(env, self.config["GLOBAL"].getfloat("learning_start"))
        print("--- Training ---")
        t_old = 0
        pbar = tqdm(total=nb_timestep, initial=1)

        while self.t <= nb_timestep:
            # tqdm stuff
            pbar.update(self.t - t_old)
            t_old, t_episode = self.t, 1

            # actual episode
            actions_taken = {action: 0 for action in range(self.action_shape)}
            done, obs, rewards = False, env.reset(), []
            actor_hidden = self.network.actor.initialize_hidden_states()
            critic_hidden = self.network.critic.initialize_hidden_states()
            reward_sum = 0
            while not done:
                action, next_actor_hidden, loss_params = self.select_action(
                    obs, actor_hidden
                )
                value, critic_hidden = self.network.get_value(obs, critic_hidden)
                next_obs, reward, done, _ = env.step(action)
                next_obs, reward = self.scaling(
                    next_obs, reward, fit=False, transform=True
                )
                next_critic_hidden = critic_hidden.copy()
                next_value, critic_hidden = self.network.get_value(
                    next_obs, critic_hidden
                )
                advantage = reward + next_value - value
                actions_taken[int(action)] += 1
                reward_sum += reward

                self.network.update_policy(advantage, *loss_params, finished=True)
                self.t, t_episode = self.t + 1, t_episode + 1
                obs = next_obs
                critic_hidden, actor_hidden = next_critic_hidden, next_actor_hidden

            artifact = self.save_if_best(reward_sum)
            if self.early_stopping(reward_sum):
                break

            self.old_reward_sum, self.episode = reward_sum, self.episode + 1
            self.episode_logging(rewards, reward_sum, actions_taken)

        pbar.close()
        self.train_logging(artifact)

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
        self.pre_train(env, self.config["GLOBAL"].getfloat("learning_start"))
        print("--- Training ---")
        t_old = 0
        pbar = tqdm(total=nb_timestep, initial=1)

        while self.t <= nb_timestep:
            # tqdm stuff
            pbar.update(self.t - t_old)
            t_old, t_episode = self.t, 1

            # actual episode
            actions_taken = {action: 0 for action in range(self.action_shape)}
            done, obs, rewards = False, env.reset(), []
            hidden = self.network.get_initial_states()
            reward_sum = 0
            while not done:
                action, next_hidden, loss_params = self.select_action(obs, hidden)
                actions_taken[int(action)] += 1
                next_obs, reward, done, _ = env.step(action)
                reward_sum += reward
                next_obs, reward = self.scaling(
                    next_obs, reward, fit=False, transform=True
                )
                self.rollout.add(reward[0], done[0], *loss_params)
                if self.rollout.full:
                    next_val, next_hidden = self.compute_value(next_obs, hidden)
                    self.rollout.update_advantages(next_val)
                    self._learn()
                    if self.rollout.done:
                        self.rollout.reset()
                    else:
                        self.rollout.clean()
                self.t, t_episode = self.t + 1, t_episode + 1
                obs, hidden = next_obs, next_hidden
                # if done:
                #     last_val = self.compute_value(next_obs, next_hidden)
            self.rollout.reset()
            # Track best model and save it
            artifact = self.save_if_best(reward_sum)
            if self.early_stopping(reward_sum):
                break

            self.old_reward_sum, self.episode = reward_sum, self.episode + 1
            self.episode_logging(rewards, reward_sum, actions_taken)

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
        print("--- Testing ---")
        if scaler_file is not None and self.obs_scaler is not None:
            with open(scaler_file, "rb") as input_file:
                scaler = pickle.load(input_file)
            self.obs_scaler = scaler
        episode_rewards = []
        best_test_episode_reward = 0
        # Iterate over the episodes
        for episode in tqdm(range(nb_episodes)):
            actor_hidden = self.network.actor.initialize_hidden_states()
            # Init episode
            done, obs, rewards_sum = False, env.reset(), 0

            # Generate episode
            while not done:
                # Select the action using the current policy
                if self.config["GLOBAL"].getboolean("scaling"):
                    obs = self.obs_scaler.transform(obs)
                action, next_actor_hidden, _ = self.select_action(obs, actor_hidden)

                # Step the environment accordingly
                next_obs, reward, done, _ = env.step(action)

                # Log reward for performance tracking
                rewards_sum += reward

                # render the environment
                if render:
                    env.render()

                # Next step
                obs, actor_hidden = next_obs, next_actor_hidden

            if rewards_sum > best_test_episode_reward:
                best_test_episode_reward = rewards_sum
                if self.config["GLOBAL"]["logging"] == "wandb":
                    wandb.run.summary["Test/best reward sum"] = rewards_sum
            # Logging
            if self.config["GLOBAL"]["logging"] == "wandb":
                wandb.log(
                    {"Test/reward": rewards_sum, "Test/episode": episode}, commit=True
                )
            elif self.config["GLOBAL"]["logging"] == "tensorboard":
                self.network.writer.add_scalar("Reward/test", rewards_sum, episode)
            # print(f"test number {episode} : {rewards_sum}")
            episode_rewards.append(rewards_sum)
        env.close()
        if self.config["GLOBAL"]["logging"] == "tensorboard":
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

    def _learn(self) -> None:
        for i, steps in enumerate(self.rollout.get_steps_list()):
            advantage, log_prob, entropy, kl_divergence = steps
            self.network.update_policy(
                advantage,
                log_prob,
                entropy,
                kl_divergence,
                finished=True,
            )


class TorchA2C(BaseTorchAgent):
    def __init__(self, agent) -> None:
        super(TorchA2C, self).__init__(agent)

        self.actor = ActorCriticRecurrentNetworks(
            agent.obs_shape[0],
            agent.action_shape,
            self.config["NETWORKS"]["actor_nn_architecture"],
            actor=True,
        )

        self.critic = ActorCriticRecurrentNetworks(
            agent.obs_shape[0],
            1,
            self.config["NETWORKS"]["critic_nn_architecture"],
            actor=False,
        )

        # Optimize to use for weight update (SGD seems to work poorly, switching to RMSProp) given our learning rate
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config["NETWORKS"].getfloat("learning_rate"),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config["NETWORKS"].getfloat("learning_rate"),
        )

        self.target_var_scaler = SimpleStandardizer()
        self.advantages_var_scaler = SimpleStandardizer()
        self.lr_scheduler = LinearSchedule(
            self.config["NETWORKS"].getfloat("learning_rate"),
            self.config["NETWORKS"].getfloat("learning_rate_end"),
            self.config["GLOBAL"].getfloat("nb_timesteps_train"),
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

    def select_action(
        self, observation: np.ndarray, hidden: Dict[int, torch.Tensor]
    ) -> np.ndarray:
        probs, new_hidden = self.actor(t(observation), hidden)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        log_prob = dist.log_prob(action.detach())
        entropy = dist.entropy()

        if self.old_dist is not None:
            KL_divergence = compute_KL_divergence(self.old_dist, dist)
        else:
            KL_divergence = 0
        action = action.flatten()[0]
        return (
            action.detach().data.numpy(),
            new_hidden,
            (log_prob, entropy, KL_divergence),
        )

    def update_policy(
        self,
        advantage: float,
        log_prob: float,
        entropy: float,
        kl_divergence: float,
        finished: bool = False,
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
        torch.autograd.set_detect_anomaly(True)
        self.index += 1
        if self.config["NETWORKS"].getboolean("normalize_advantages"):
            advantages = torch.div(
                torch.sub(advantages, advantages.mean()),
                torch.add(advantages.std(), 1e-8),
            )

        # Losses (be careful that all its components are torch tensors with grad on)
        entropy_loss = -entropy
        actor_loss = -(log_prob * advantage.detach())
        critic_loss = advantage.pow(2)
        kl_loss = -kl_divergence
        actor_loss = (
            actor_loss
            + self.config["AGENT"].getfloat("entropy_factor") * entropy_loss
            # + self.config["AGENT"].getfloat("KL_factor") * kl_loss
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.gradient_clipping()  #
        if finished:
            self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        # self.gradient_clipping()
        if finished:
            self.critic_optimizer.step()

        # KPIs
        # explained_variance = self.compute_explained_variance(
        #     empirical_return.detach().numpy(),
        #     advantages.detach().numpy(),
        # )
        # self.old_dist = dist

        # Logging

        if self.config["GLOBAL"]["logging"] == "tensorboard":
            if self.writer:
                self.writer.add_scalar("Train/entropy loss", -entropy_loss, self.index)
                self.writer.add_scalar(
                    "Train/leaarning rate",
                    self.lr_scheduler.transform(self.index),
                    self.index,
                )
                self.writer.add_scalar("Train/policy loss", actor_loss, self.index)
                self.writer.add_scalar("Train/critic loss", critic_loss, self.index)
                # self.writer.add_scalar(
                #     "Train/explained variance", explained_variance, self.index
                # )
                # self.writer.add_scalar("Train/kl divergence", KL_divergence, self.index)
            else:
                warnings.warn("No Tensorboard writer available")
        elif self.config["GLOBAL"]["logging"] == "wandb":
            wandb.log(
                {
                    "Train/entropy loss": -entropy_loss,
                    "Train/actor loss": actor_loss,
                    "Train/critic loss": critic_loss,
                    # "Train/explained variance": explained_variance,
                    # "Train/KL divergence": KL_divergence,
                    "Train/learning rate": self.lr_scheduler.transform(self.index),
                },
                commit=False,
            )

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
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

    def get_value(
        self, state: np.ndarray, hidden: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Computes the state value for the given state s and for the current policy parameters theta.
        Same as forward method with a clearer name for teaching purposes, but as forward is a native method that needs to exist we keep both.
        Additionnaly this methods outputs np.array instead of torch.Tensor to prevent the existence of pytorch stuff outside of network.py

        Args:
            state (np.array): np.array representation of the state

        Returns:
            np.array: np.array representation of the action probabilities
        """
        value, new_hidden = self.critic(t(state), hidden)
        return value, new_hidden

    def save(self, name: str = "model") -> None:
        """
        Save the current model

        Args:
            name (str, optional): [Name of the model]. Defaults to "model".
        """
        torch.save(self.actor, f'{self.config["PATHS"]["model_path"]}/{name}_actor.pth')
        torch.save(
            self.critic, f'{self.config["PATHS"]["model_path"]}/{name}_critic.pth'
        )

    def load(self, name: str = "model") -> None:
        """
        Load the designated model

        Args:
            name (str, optional): The model to be loaded (it should be in the "models" folder). Defaults to "model".
        """
        print("Loading")
        self.actor = torch.load(
            f'{self.config["PATHS"]["model_path"]}/actor_{name}.pth'
        )
        self.critic = torch.load(
            f'{self.config["PATHS"]["model_path"]}/critic_{name}.pth'
        )

    def fit_transform(self, input) -> torch.Tensor:
        self.scaler.partial_fit(input)
        if self.index > 2:
            return t(self.scaler.transform(input))

    def gradient_clipping(self) -> None:
        clip_value = self.config["AGENT"].getfloat("gradient_clipping")
        if clip_value is not None:
            for optimizer in [self.actor_optimizer, self.critic_optimizer]:
                nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    clip_value,
                )

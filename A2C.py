import pickle

import gym
import wandb
from pymgrid.Environments.MacroEnvironment import RBCPolicy
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Network creator tool
from network_utils import t, compute_KL_divergence
from utils import LinearSchedule
from normalize import SimpleStandardizer

# Base class for Agent
from agent import Agent
from buffer import RolloutBuffer
import warnings

# The network we create and the device to run it on
from network import ActorCriticRecurrentNetworks, BaseTorchAgent


# Numpy
import numpy as np

# For logging
from tqdm import tqdm
import wandb


class A2C(Agent):
    def __init__(self, env: gym.Env, config: dict, comment: str = "", run=None) -> None:
        super(
            A2C,
            self,
        ).__init__(env, config, comment, run)

        self.rollout = RolloutBuffer(
            buffer_size=config["BUFFER_SIZE"],
            gamma=config["GAMMA"],
            n_steps=config["N_STEPS"],
        )

        self.obs_scaler, self.reward_scaler, self.target_scaler = self.get_scalers()

        # Initialize the policy network with the right shape
        self.network = TorchA2C(self)
        self.network.to(self.network.device)
        self.network.writer = SummaryWriter(log_dir=self.log_dir)

    def select_action(
        self,
        observation: np.array,
        hidden: np.array,
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
        self.t, t_old, self.constant_reward_counter = 1, 0, 0

        # Pre-Training
        if self.config["LEARNING_START"] > 0:
            print("--- Pre-Training ---")
            t_pre_train = 1
            pbar = tqdm(total=self.config["LEARNING_START"], initial=1)
            while t_pre_train <= self.config["LEARNING_START"]:
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
                self.rollout.add(reward, done, *loss_params)
                if self.rollout.full:
                    next_val, next_hidden = self.compute_value(next_obs, hidden)
                    self.rollout.update_advantages(next_val)
                    self._learn()
                    self.rollout.reset()
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
            hidden = self.network.get_initial_states()
            # Init episode
            done, obs, rewards_sum = False, env.reset(), 0

            # Generate episode
            while not done:
                # Select the action using the current policy
                if self.config["SCALING"]:
                    obs = self.obs_scaler.transform(obs)
                action, next_hidden, _ = self.select_action(obs, hidden)

                # Step the environment accordingly
                next_obs, reward, done, _ = env.step(action)

                # Log reward for performance tracking
                rewards_sum += reward

                # render the environment
                if render:
                    env.render()

                # Next step
                obs, hidden = next_obs, next_hidden

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

    def _learn(self):
        for i, steps in enumerate(self.rollout.get_steps_list()):
            advantage, log_prob, entropy, kl_divergence = steps
            self.network.update_policy(
                advantage,
                log_prob,
                entropy,
                kl_divergence,
                finished=i == self.config["BUFFER_SIZE"] - 1,
            )


class TorchA2C(BaseTorchAgent):
    def __init__(self, agent):
        super(TorchA2C, self).__init__(agent)

        # Create the network architecture given the observation, action and config shapes
        self.actorcritic = ActorCriticRecurrentNetworks(
            agent.obs_shape[0],
            agent.action_shape,
            self.config["HIDDEN_SIZE"],
            self.config["HIDDEN_LAYERS"],
            self.config,
        )
        print(self.actorcritic)

        if self.config["logging"] == "wandb":
            wandb.watch(self.actorcritic)

        # Optimize to use for weight update (SGD seems to work poorly, switching to RMSProp) given our learning rate
        self.optimizer = torch.optim.Adam(
            self.actorcritic.parameters(),
            lr=self.config["LEARNING_RATE"],
        )
        # self.critic_optimizer = optim.Adam(
        #     self.actorcritic.critic_layers.parameters(),
        #     lr=self.config["LEARNING_RATE"],
        # )

        self.target_var_scaler = SimpleStandardizer()
        self.advantages_var_scaler = SimpleStandardizer()
        self.lr_scheduler = LinearSchedule(
            self.config["LEARNING_RATE"],
            self.config["LEARNING_RATE_END"],
            self.config["NB_TIMESTEPS_TRAIN"],
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

    def get_latent_observation(
        self, observation: np.array, hidden: np.array
    ) -> torch.Tensor:
        observation = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)
        embedded_observation = self.actorcritic.forward(observation)

        if self.config["RECURRENT"]:
            new_embedded_observation, new_hidden = self.actorcritic.LSTM(
                embedded_observation, hidden
            )
        else:
            new_embedded_observation = embedded_observation
            new_hidden = None

        return embedded_observation, new_hidden

    def select_action(self, observation: np.array, hidden: np.array) -> np.array:
        """Select the action based on the observation and the current parametrized policy

        Args:
            observation (np.array): The current state observation

        Returns:
            np.array : The selected action(s)
        """
        embedded_observation, new_hidden = self.get_latent_observation(
            observation, hidden
        )
        dist = self.actorcritic.actor(embedded_observation)
        value = self.actorcritic.critic(embedded_observation)

        action = dist.sample()
        log_prob = dist.log_prob(action.detach())
        entropy = dist.entropy()
        if self.old_dist is not None:
            KL_divergence = compute_KL_divergence(self.old_dist, dist)
        else:
            KL_divergence = 0
        loss_params = (value, log_prob, entropy, KL_divergence)

        if self.continuous:
            if self.config["LAW"] == "normal":
                action = np.clip(
                    action, self.env.action_space.low, self.env.action_space.high
                )
            elif self.config["LAW"] == "beta":
                action = (
                    action * (self.env.action_space.high - self.env.action_space.low)
                    + self.env.action_space.low
                )
            return action.detach().data.numpy()[0][0], new_hidden
        else:
            assert len(action) == 1, "Bug action"
            action = action.flatten()[0]
            return action.detach().data.numpy(), new_hidden, loss_params

    def update_policy(
        self, advantage, log_prob, entropy, kl_divergence, finished=False
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
        # self.optimizer = optim.Adam(
        #     self.actorcritic.parameters(),
        #     lr=self.lr_scheduler.transform(self.index),
        # )
        # if self.config["NORMALIZE_ADVANTAGES"]:
        #     advantages = torch.div(
        #         torch.sub(advantages, advantages.mean()),
        #         torch.add(advantages.std(), 1e-8),
        #     )

        # Losses (be careful that all its components are torch tensors with grad on)
        entropy_loss = -entropy
        actor_loss = -(log_prob * advantage.detach())
        critic_loss = advantage.pow(2)
        kl_loss = -kl_divergence
        loss = (
            actor_loss
            + self.config["VALUE_FACTOR"] * critic_loss
            + self.config["ENTROPY_FACTOR"] * entropy_loss
            # + self.config["KL_FACTOR"] * kl_loss
        )
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if self.config["GRADIENT_CLIPPING"] is not None:
            self.gradient_clipping()
        if finished:
            self.optimizer.step()

        # KPIs
        # explained_variance = self.compute_explained_variance(
        #     empirical_return.detach().numpy(),
        #     advantages.detach().numpy(),
        # )
        # self.old_dist = dist

        # Logging
        if self.writer:
            if self.config["logging"] == "tensorboard":
                self.writer.add_scalar("Train/entropy loss", -entropy_loss, self.index)
                self.writer.add_scalar(
                    "Train/leaarning rate",
                    self.lr_scheduler.transform(self.index),
                    self.index,
                )
                self.writer.add_scalar("Train/policy loss", actor_loss, self.index)
                self.writer.add_scalar("Train/critic loss", critic_loss, self.index)
                self.writer.add_scalar("Train/total loss", loss, self.index)
                # self.writer.add_scalar(
                #     "Train/explained variance", explained_variance, self.index
                # )
                # self.writer.add_scalar("Train/kl divergence", KL_divergence, self.index)
        else:
            warnings.warn("No Tensorboard writer available")
        if self.config["logging"] == "wandb":
            wandb.log(
                {
                    "Train/entropy loss": -entropy_loss,
                    "Train/actor loss": actor_loss,
                    "Train/critic loss": critic_loss,
                    "Train/total loss": loss,
                    # "Train/explained variance": explained_variance,
                    # "Train/KL divergence": KL_divergence,
                    "Train/learning rate": self.lr_scheduler.transform(self.index),
                },
                commit=False,
            )

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
        return self.actorcritic.actor(t(state)).detach().cpu().numpy()

    def get_value(self, state: np.array, hidden) -> np.array:
        """
        Computes the state value for the given state s and for the current policy parameters theta.
        Same as forward method with a clearer name for teaching purposes, but as forward is a native method that needs to exist we keep both.
        Additionnaly this methods outputs np.array instead of torch.Tensor to prevent the existence of pytorch stuff outside of network.py

        Args:
            state (np.array): np.array representation of the state

        Returns:
            np.array: np.array representation of the action probabilities
        """
        embedded_observation, new_hidden = self.get_latent_observation(state, hidden)
        return (
            self.actorcritic.critic(embedded_observation),
            new_hidden,
        )

    def save(self, name: str = "model"):
        """
        Save the current model

        Args:
            name (str, optional): [Name of the model]. Defaults to "model".
        """
        torch.save(self.actorcritic, f'{self.config["MODEL_PATH"]}/{name}.pth')

    def load(self, name: str = "model"):
        """
        Load the designated model

        Args:
            name (str, optional): The model to be loaded (it should be in the "models" folder). Defaults to "model".
        """
        print("Loading")
        self.actorcritic = torch.load(f'{self.config["MODEL_PATH"]}/{name}.pth')
        self.actorcritic.config = self.config

        print(self.actorcritic)

    def get_initial_states(self):
        if self.config["RECURRENT"]:
            h_0, c_0 = None, None

            h_0 = torch.zeros(
                (
                    self.actorcritic._recurrent_layer.num_layers,
                    1,
                    self.actorcritic._recurrent_layer.hidden_size,
                ),
                dtype=torch.float,
            )
            h_0 = h_0.to(device=self.device)

            c_0 = torch.zeros(
                (
                    self.actorcritic._recurrent_layer.num_layers,
                    1,
                    self.actorcritic._recurrent_layer.hidden_size,
                ),
                dtype=torch.float,
            )
            c_0 = c_0.to(device=self.device)
            return (h_0, c_0)
        else:
            return None

    def fit_transform(self, input):
        self.scaler.partial_fit(input)
        if self.index > 2:
            return t(self.scaler.transform(input))

    def gradient_clipping(self):
        nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            self.config["GRADIENT_CLIPPING"],
        )  # gradient clipping

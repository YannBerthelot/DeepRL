from agent import Agent
from network import PolicyNetwork, device
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import Config
from tqdm import tqdm

writer = SummaryWriter()


class REINFORCE(Agent):
    def __init__(self, env) -> None:
        super(Agent, self).__init__()
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.n
        self.policy_network = PolicyNetwork(self.obs_shape, self.action_shape).to(
            device
        )
        self.global_idx_episode = 0
        self.best_episode_reward = 0

    def collect_rollout(self, env, nb_episodes):
        rollout = []
        rewards = []

        for idx_episode in range(nb_episodes):
            episode = []
            reward_sum = 0
            obs = env.reset()
            done = False
            t = 0
            while not done:
                action = self.select_action(obs)
                next_obs, reward, done, _ = env.step(action)
                reward_sum += reward
                step = {"timestep": t, "obs": obs, "action": action, "reward": reward}
                obs = next_obs
                episode.append(step)
                t += 1
            rollout.append(episode)
            rewards.append(reward_sum)
            if reward_sum > self.best_episode_reward:
                self.best_episode_reward = reward_sum
                self.save("best")
            self.global_idx_episode += 1
            writer.add_scalar(
                "Reward/collection", np.mean(rewards), self.global_idx_episode
            )
        return rollout, rewards

    def train(self, env, nb_episodes_per_epoch, nb_epoch):
        self.global_idx_episode = 0
        for epoch in tqdm(range(nb_epoch)):
            rollout, rewards = self.collect_rollout(env, nb_episodes_per_epoch)
            # print(f"epoch {epoch} : {np.sum(rewards)}")
            for episode in rollout:
                episode = self.compute_return(episode, gamma=Config.GAMMA)
                for timestep in episode:
                    self.policy_network.update_policy(
                        timestep["obs"], timestep["action"], timestep["return"]
                    )

    def select_action(self, observation):
        action_probabilities = self.policy_network.select_action(observation)
        action = np.random.choice(
            list(range(self.action_shape)), p=action_probabilities
        )
        return int(action)

    def compute_return(self, episode, gamma=0.99):
        if not (0 <= gamma <= 1):
            raise ValueError(f"Gamma not between 0 and 1, gamma = {gamma}")
        G = 0
        for i, step in enumerate(reversed(episode)):
            G = step["reward"] + (gamma ** i) * G
            step["return"] = G
            episode[len(episode) - i - 1] = step
        return episode

    def test(self, env, nb_episodes, render=False):
        for episode in range(nb_episodes):
            done = False
            obs = env.reset()
            rewards_sum = 0
            while not done:
                action = self.select_action(obs)
                next_obs, reward, done, _ = env.step(action)
                rewards_sum += 1
                obs = next_obs
                if render:
                    env.render()
            writer.add_scalar("Reward/test", rewards_sum, episode)
            print(f"test number {episode} : {rewards_sum}")

    def save(self, name="model"):
        self.policy_network.save(name)

    def load(self, name):
        self.policy_network.load(name)

import os
import gym
import unittest
from deeprlyb.agents.A2C import A2C
from deeprlyb.utils.config import read_config


class TestA2C(unittest.TestCase):
    def test_init(self) -> None:
        env = gym.make("CartPole-v1")
        dir = os.path.dirname(__file__)
        config_file = os.path.join(dir, "config.ini")
        config = read_config(config_file)
        agent = A2C(env, config)

    def test_networks(self) -> None:
        env = gym.make("CartPole-v1")
        dir = os.path.dirname(__file__)
        config_file = os.path.join(dir, "config.ini")
        config = read_config(config_file)
        agent = A2C(env, config)
        obs = env.reset()
        actor_hidden = agent.network.actor.initialize_hidden_states()
        critic_hidden = agent.network.critic.initialize_hidden_states()
        for i in range(10):
            action, actor_hidden = agent.select_action(obs, actor_hidden)[:2]
            value, critic_hidden = agent.network.get_value(obs, critic_hidden)
            obs, reward, done, _ = env.step(action)
            if done:
                break
        config["NETWORKS"]["actor_nn_architecture"] = "[128,LSTM(16),LSTM(8),256]"
        config["NETWORKS"]["critic_nn_architecture"] = "[128,LSTM(16),LSTM(8),256]"
        agent = A2C(env, config)
        obs = env.reset()
        actor_hidden = agent.network.actor.initialize_hidden_states()
        critic_hidden = agent.network.critic.initialize_hidden_states()
        for i in range(10):
            action, actor_hidden = agent.select_action(obs, actor_hidden)[:2]
            value, critic_hidden = agent.network.get_value(obs, critic_hidden)
            obs, reward, done, _ = env.step(action)
            if done:
                break

    def test_update(self) -> None:
        env = gym.make("CartPole-v1")
        dir = os.path.dirname(__file__)
        config_file = os.path.join(dir, "config.ini")
        config = read_config(config_file)
        agent = A2C(env, config)
        obs = env.reset()
        actor_hidden = agent.network.actor.initialize_hidden_states()
        critic_hidden = agent.network.critic.initialize_hidden_states()
        for i in range(10):
            action, actor_hidden, loss_params = agent.select_action(obs, actor_hidden)
            value, critic_hidden = agent.network.get_value(obs, critic_hidden)
            obs, reward, done, _ = env.step(action)
            next_value, critic_hidden = agent.network.get_value(obs, critic_hidden)
            advantage = reward + next_value - value
            agent.network.update_policy(advantage, *loss_params)

            if done:
                break

    def test_train_test(self) -> None:
        env = gym.make("CartPole-v1")
        dir = os.path.dirname(__file__)
        config_file = os.path.join(dir, "config.ini")
        config = read_config(config_file)
        agent = A2C(env, config)
        agent.train_TD0(env, 1e3)
        agent.test(env, nb_episodes=10, render=False)


if __name__ == "__main__":
    unittest.main()

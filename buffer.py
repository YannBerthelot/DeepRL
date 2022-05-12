import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, buffer_size, gamma, n_steps) -> None:
        self._n_steps = n_steps
        self._buffer_size = buffer_size
        self.gamma = gamma

        self.reset()

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def __len__(self):
        return len(self.rewards)

    def add(self, reward, done, value, log_prob, entropy, KL_divergence):
        self.rewards = np.append(self.rewards, reward)
        self.dones = np.append(self.dones, done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.KL_divergences = np.append(self.KL_divergences, KL_divergence)
        if self.__len__ >= self.buffer_size + self.n_steps - 1:
            self.full = True

    @staticmethod
    def compute_returns_and_advantages(
        rewards, values, dones, gamma, last_val, buffer_size, n_steps
    ):
        next_values = values[n_steps:]
        next_values.append(last_val)
        n_step_return = 0
        returns = []
        for j in range(buffer_size):
            rewards_list = rewards[j : j + n_steps]
            for i, reward in enumerate(reversed(rewards_list)):
                n_step_return = reward + (gamma ** i) * n_step_return
            returns.append(n_step_return)
        returns = returns[::-1].copy()
        if n_steps > 0:
            return [
                returns[i]
                + (1 - dones[i]) * (gamma ** n_steps) * next_values[i]
                - values[i]
                for i in range(len(values) - n_steps + 1)
            ]

    def update_advantages(self, last_val: float):
        """Wrapper of static method compute_returns_and_advantages

        Args:
            last_val (float): The last value computed by the value network
        """
        self.advantages = self.compute_returns_and_advantages(
            self.rewards,
            self.values,
            self.dones,
            self.gamma,
            last_val,
            self.buffer_size,
            self.n_steps,
        )

    def show(self):
        print("ADVANTAGES", self.advantages)
        print("LOG PROBS", self.log_probs)
        print("ENTROPIES", self.entropies)
        print("KL_DIVERGENCES", self.KL_divergences)

    def get_steps(self):
        return self.advantages, self.log_probs, self.entropies, self.KL_divergences

    def get_steps_list(self):
        for i in range(len(self.advantages)):
            yield self.advantages[i], self.log_probs[i], self.entropies[
                i
            ], self.KL_divergences[i]

    def clear(self):
        print(self.values)
        self.rewards = self.rewards[self.buffer_size :]
        self.dones = self.dones[self.buffer_size :]
        self.KL_divergences = self.KL_divergences[self.buffer_size :]
        self.values = self.values[self.buffer_size :]
        self.log_probs = self.log_probs[self.buffer_size :]
        self.entropies = self.entropies[self.buffer_size :]
        print(self.values)

    def reset(self):
        self.rewards = np.array([])
        self.dones = np.array([])
        self.KL_divergences = np.array([])
        self.values = []
        self.log_probs = []
        self.entropies = []

        self.rewards = np.array([])
        self.dones = np.array([])
        self.KL_divergences = np.array([])
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.full = False


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
        for i, reward in enumerate(reversed(self.steps["rewards"][:-1])):
            n_step_return = (
                reward
                + 1  # (1.0 - self.steps["dones"][i])
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

    def show(self):
        print(self.steps)

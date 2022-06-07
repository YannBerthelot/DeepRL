import numpy as np
import numpy.typing as npt
from typing import Union
from functools import reduce


class RolloutBuffer:
    def __init__(
        self, buffer_size: int, gamma: float, n_steps: int, setting: str = "MC"
    ) -> None:
        if buffer_size < 1:
            raise ValueError("Buffer size must be positive")
        if not (0 <= gamma and gamma <= 1):
            raise ValueError("Gamma must be between 0 and 1")
        if n_steps < 1:
            raise ValueError("Number of steps must be positive")
        self.setting = setting
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
    def done(self):
        return max(self.dones) == 1

    @property
    def full(self):
        if (self.__len__ >= self.buffer_size + self.n_steps - 1) or self.done:
            return True
        else:
            return False

    def reset(self):
        buffer_size = self.buffer_size + self.n_steps - 1
        self.rewards = np.zeros(buffer_size)
        self.dones = np.zeros(buffer_size)
        self.KL_divergences = np.zeros(buffer_size)
        self.values = np.zeros(buffer_size)
        self.log_probs = np.zeros(buffer_size)
        self.entropies = np.zeros(buffer_size)
        self.advantages = None
        self._returns = None
        self.__len__ = 0

    def clean(self):
        buffer_size = self.buffer_size + self.n_steps - 1
        old_rewards = self.rewards[-self.n_steps + 1 :]
        self.rewards = np.zeros(buffer_size)
        self.rewards[: self.n_steps - 1] = old_rewards
        self.dones = np.zeros(buffer_size)
        self.KL_divergences = np.zeros(buffer_size)
        self.values = np.zeros(buffer_size)
        self.log_probs = np.zeros(buffer_size)
        self.entropies = np.zeros(buffer_size)
        self.advantages = None
        self._returns = None
        self.__len__ = self.n_steps - 1

    def add(
        self,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        entropy: float,
        KL_divergence: float,
    ):
        self.dones[self.__len__] = done
        self.values[self.__len__] = value
        self.log_probs[self.__len__] = log_prob
        self.entropies[self.__len__] = entropy
        self.KL_divergences[self.__len__] = KL_divergence
        self.rewards[self.__len__] = reward
        self.__len__ += 1

    @staticmethod
    def compute_n_step_return(rewards: npt.NDArray[np.float64], gamma: float):
        # return reduce(lambda a, b: gamma * a + b, reversed(rewards)) -> less efficient somehow
        n_step_return = 0
        for reward in reversed(rewards):
            n_step_return = reward + gamma * n_step_return
        return n_step_return

    @staticmethod
    def compute_next_return(
        last_return: float, R_0: float, R_N: float, gamma: float, n_steps: int
    ):
        return ((last_return - R_0) / gamma) + ((gamma ** n_steps) * R_N)

    @staticmethod
    def compute_all_n_step_returns(
        rewards: npt.NDArray[np.float64],
        gamma: float,
        buffer_size: int,
        n_steps: int,
    ):
        returns = []
        for j in range(buffer_size):
            rewards_list = rewards[j : min(j + n_steps, len(rewards))]
            returns.append(RolloutBuffer.compute_n_step_return(rewards_list, gamma))
        return returns

    @staticmethod
    def compute_returns(
        rewards: npt.NDArray[np.float64],
        gamma: float,
        buffer_size: int,
        n_steps: int,
    ):

        returns = []

        # Compute initial return
        G_0 = RolloutBuffer.compute_n_step_return(rewards[0 : 0 + n_steps], gamma)
        returns.append(G_0)
        for i in range(buffer_size - 1):
            new_return = RolloutBuffer.compute_next_return(
                returns[i], rewards[i], rewards[i + n_steps], gamma, n_steps - 1
            )
            returns.append(new_return)

        returns = np.array(returns)
        return returns

    @staticmethod
    def compute_advantages(
        returns: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        dones: npt.NDArray[np.float64],
        gamma: float,
        last_val: float,
        n_steps: int,
    ):

        next_values = values[n_steps:]
        next_values = np.append(next_values, last_val)

        if n_steps > 0:
            return [
                returns[i]
                + (1 - dones[i]) * (gamma ** n_steps) * next_values[i]
                - values[i]
                for i in range(len(values) - n_steps + 1)
            ]
        else:
            raise ValueError(f"Invalid steps number : {n_steps}")

    def update_advantages(self, last_val: float, fast="True"):
        """Wrapper of static method compute_returns_and_advantages

        Args:
            last_val (float): The last value computed by the value network
        """
        if (self.n_steps > 2) and fast:
            # Faster for high number of steps (complexity constant with n-steps)
            self._returns = RolloutBuffer.compute_returns(
                self.rewards,
                self.gamma,
                self.buffer_size,
                self.n_steps,
            )

        else:
            # Faster for low number of steps
            self._returns = RolloutBuffer.compute_all_n_step_returns(
                self.rewards,
                self.gamma,
                self.buffer_size,
                self.n_steps,
            )

        self.advantages = RolloutBuffer.compute_advantages(
            self._returns, self.values, self.dones, self.gamma, last_val, self.n_steps
        )

    def show(self):
        print("REWARDS", self.rewards)
        print("VALUES", self.values)
        print("DONES", self.dones)
        print("LOG PROBS", self.log_probs)
        print("ENTROPIES", self.entropies)
        print("KL_DIVERGENCES", self.KL_divergences)
        if self.advantages is not None:
            print("ADVANTAGES", self.advantages)

    def get_steps(self):
        return self.advantages, self.log_probs, self.entropies, self.KL_divergences

    def get_steps_list(self):
        for i in range(len(self.advantages)):
            yield self.advantages[i], self.log_probs[i], self.entropies[
                i
            ], self.KL_divergences[i]

    def clear(self):
        self.rewards = self.rewards[self.buffer_size :]
        self.dones = self.dones[self.buffer_size :]
        self.KL_divergences = self.KL_divergences[self.buffer_size :]
        self.values = self.values[self.buffer_size :]
        self.log_probs = self.log_probs[self.buffer_size :]
        self.entropies = self.entropies[self.buffer_size :]


class RolloutBufferLambda:
    def __init__(
        self, buffer_size: int, gamma: float, n_steps: int, setting: str = "MC"
    ) -> None:
        if buffer_size < 1:
            raise ValueError("Buffer size must be positive")
        if not (0 <= gamma and gamma <= 1):
            raise ValueError("Gamma must be between 0 and 1")
        if n_steps < 1:
            raise ValueError("Number of steps must be positive")
        self.setting = setting
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
    def done(self):
        return max(self.dones) == 1

    @property
    def full(self):
        if (self.__len__ >= self.buffer_size + self.n_steps - 1) or self.done:
            return True
        else:
            return False

    def reset(self):
        buffer_size = self.buffer_size + self.n_steps - 1
        self.observations = np.zeros(buffer_size)
        self.rewards = np.zeros(buffer_size)
        self.dones = np.zeros(buffer_size)
        self.KL_divergences = np.zeros(buffer_size)
        self.values = np.zeros(buffer_size)
        self.log_probs = np.zeros(buffer_size)
        self.entropies = np.zeros(buffer_size)
        self.advantages = None
        self._returns = None
        self.__len__ = 0

    def clean(self):
        buffer_size = self.buffer_size + self.n_steps - 1
        old_rewards = self.rewards[-self.n_steps + 1 :]
        self.rewards = np.zeros(buffer_size)
        self.rewards[: self.n_steps - 1] = old_rewards
        self.dones = np.zeros(buffer_size)
        self.KL_divergences = np.zeros(buffer_size)
        self.values = np.zeros(buffer_size)
        self.log_probs = np.zeros(buffer_size)
        self.entropies = np.zeros(buffer_size)
        self.advantages = None
        self._returns = None
        self.__len__ = self.n_steps - 1

    def add(
        self,
        observation: npt.NDArray[np.float64],
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        entropy: float,
        KL_divergence: float,
    ):
        self.observations[self.__len__] = observation
        self.dones[self.__len__] = done
        self.values[self.__len__] = value
        self.log_probs[self.__len__] = log_prob
        self.entropies[self.__len__] = entropy
        self.KL_divergences[self.__len__] = KL_divergence
        self.rewards[self.__len__] = reward
        self.__len__ += 1

    @staticmethod
    def compute_n_step_return(rewards: npt.NDArray[np.float64], gamma: float):
        # return reduce(lambda a, b: gamma * a + b, reversed(rewards)) -> less efficient somehow
        n_step_return = 0
        for reward in reversed(rewards):
            n_step_return = reward + gamma * n_step_return
        return n_step_return

    @staticmethod
    def compute_next_return(
        last_return: float, R_0: float, R_N: float, gamma: float, n_steps: int
    ):
        return ((last_return - R_0) / gamma) + ((gamma ** n_steps) * R_N)

    @staticmethod
    def compute_all_n_step_returns(
        rewards: npt.NDArray[np.float64],
        gamma: float,
        buffer_size: int,
        n_steps: int,
    ):
        returns = []
        for j in range(buffer_size):
            rewards_list = rewards[j : min(j + n_steps, len(rewards))]
            returns.append(RolloutBuffer.compute_n_step_return(rewards_list, gamma))
        return returns

    @staticmethod
    def compute_returns(
        rewards: npt.NDArray[np.float64],
        gamma: float,
        buffer_size: int,
        n_steps: int,
    ):

        returns = []

        # Compute initial return
        G_0 = RolloutBuffer.compute_n_step_return(rewards[0 : 0 + n_steps], gamma)
        returns.append(G_0)
        for i in range(buffer_size - 1):
            new_return = RolloutBuffer.compute_next_return(
                returns[i], rewards[i], rewards[i + n_steps], gamma, n_steps - 1
            )
            returns.append(new_return)

        returns = np.array(returns)
        return returns

    @staticmethod
    def compute_advantages(
        returns: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        dones: npt.NDArray[np.float64],
        gamma: float,
        last_val: float,
        n_steps: int,
    ):

        next_values = values[n_steps:]
        next_values = np.append(next_values, last_val)

        if n_steps > 0:
            return [
                returns[i]
                + (1 - dones[i]) * (gamma ** n_steps) * next_values[i]
                - values[i]
                for i in range(len(values) - n_steps + 1)
            ]
        else:
            raise ValueError(f"Invalid steps number : {n_steps}")

    def update_advantages(self, last_val: float, fast="True"):
        """Wrapper of static method compute_returns_and_advantages

        Args:
            last_val (float): The last value computed by the value network
        """
        if (self.n_steps > 2) and fast:
            # Faster for high number of steps (complexity constant with n-steps)
            self._returns = RolloutBuffer.compute_returns(
                self.rewards,
                self.gamma,
                self.buffer_size,
                self.n_steps,
            )

        else:
            # Faster for low number of steps
            self._returns = RolloutBuffer.compute_all_n_step_returns(
                self.rewards,
                self.gamma,
                self.buffer_size,
                self.n_steps,
            )

        self.advantages = RolloutBuffer.compute_advantages(
            self._returns, self.values, self.dones, self.gamma, last_val, self.n_steps
        )

    def show(self):
        print("REWARDS", self.rewards)
        print("VALUES", self.values)
        print("DONES", self.dones)
        print("LOG PROBS", self.log_probs)
        print("ENTROPIES", self.entropies)
        print("KL_DIVERGENCES", self.KL_divergences)
        if self.advantages is not None:
            print("ADVANTAGES", self.advantages)

    def get_steps(self):
        return self.advantages, self.log_probs, self.entropies, self.KL_divergences

    def get_steps_list(self):
        for i in range(len(self.advantages)):
            yield self.advantages[i], self.log_probs[i], self.entropies[
                i
            ], self.KL_divergences[i]

    def clear(self):
        self.rewards = self.rewards[self.buffer_size :]
        self.dones = self.dones[self.buffer_size :]
        self.KL_divergences = self.KL_divergences[self.buffer_size :]
        self.values = self.values[self.buffer_size :]
        self.log_probs = self.log_probs[self.buffer_size :]
        self.entropies = self.entropies[self.buffer_size :]


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

    def __len__(self):
        return len(self.steps["rewards"])

    def show(self):
        print(self.steps)

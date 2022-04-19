import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, buffer_size, gamma, n_steps) -> None:
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.reset()

    def add(self, reward, done, value, log_prob, entropy, KL_divergence):
        self.rewards = np.append(self.rewards, reward)
        self.dones = np.append(self.dones, done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.KL_divergences = np.append(self.KL_divergences, KL_divergence)
        if self.__len__() >= self.buffer_size + self.n_steps - 1:
            self.full = True

    def compute_returns_and_advantages(self, last_val):
        self.next_values = self.values[self.n_steps :]
        self.next_values.append(last_val)
        # self.next_values = torch.cat((self.next_values, last_value), dim=0)
        n_step_return = 0
        returns = []
        for j in range(self.buffer_size):
            rewards_list = self.rewards[j : j + self.n_steps]
            for i, reward in enumerate(reversed(rewards_list)):
                n_step_return = reward + (self.gamma ** i) * n_step_return
            returns.append(n_step_return)
        returns = returns[::-1].copy()
        if self.n_steps > 0:
            self.advantages = [
                returns[i]
                + (1 - self.dones[i])
                * (self.gamma ** self.n_steps)
                * self.next_values[i]
                - self.values[i]
                for i in range(len(self.values) - self.n_steps + 1)
            ]

        # self.explained_variances = compute_explained_variance(
        #     np.array(returns),
        #     self.advantages.detach().numpy(),
        # )
        # (
        #     returns.copy()
        #     # + (1 - self.dones) * (self.gamma ** self.n_steps) * self.next_values
        #     - self.values
        # )

    def __len__(self):
        return len(self.rewards)

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

    def clear(self, val):
        self.rewards = self.rewards[self.n_steps :]
        self.dones = self.dones[self.n_steps :]
        self.KL_divergences = self.KL_divergences[self.n_steps :]
        self.values = [val]
        self.log_probs = self.log_probs[self.n_steps :]
        self.entropies = self.entropies[self.n_steps :]

    def reset(self):
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

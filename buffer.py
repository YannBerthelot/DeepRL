import numpy as np


class RolloutBuffer:
    def __init__(self, buffer_size) -> None:

        self.buffer_size = buffer_size
        self.reset()

    def add(
        self,
        state,
        next_state,
        action,
        reward,
        done,
        hidden=None,
        next_hidden=None,
    ):
        if self.states.shape[0] == 0:
            self.states = np.array([state])
            self.next_states = np.array([next_state])
            self.actions = np.array([action])
            self.rewards = np.array([reward])
            self.dones = np.array([done])
            self.hiddens = np.array([hidden])
            self.next_hiddens = np.array([next_hidden])
        else:
            self.states = np.append(self.states, np.array([state]), axis=0)
            self.next_states = np.append(
                self.next_states, np.array([next_state]), axis=0
            )
            self.actions = np.append(self.actions, np.array([action]), axis=0)
            self.rewards = np.append(self.rewards, np.array([reward]), axis=0)
            self.dones = np.append(self.dones, np.array([done]), axis=0)
            self.hiddens = np.append(self.hiddens, np.array([hidden]), axis=0)
            self.next_hiddens = np.append(
                self.next_hiddens, np.array([next_hidden]), axis=0
            )

    def reset(self):
        self.states = np.array([])
        self.next_states = np.array([])
        self.actions = np.array([])
        self.rewards = np.array([])
        self.dones = np.array([])
        self.hiddens = np.array([])
        self.next_hiddens = np.array([])
        self.returns = np.array([])

    def show(self):
        print(
            "states",
            self.states,
            "next_states",
            self.next_states,
            "actions",
            self.actions,
            "rewards",
            self.rewards,
            "dones",
            self.dones,
            "hiddens",
            self.hiddens,
            "next_hiddens",
            self.next_hiddens,
        )


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

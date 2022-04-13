import numpy as np
import torch


class SimpleMinMaxScaler:
    def __init__(self, maxs: list, mins: list, feature_range: tuple = (0, 1)) -> None:
        self.max = np.array(maxs)
        self.min = np.array(mins)
        self.feature_range = feature_range
        if feature_range[0] == feature_range[1]:
            raise ValueError(
                f"Feature range values must be different ({feature_range[0]} and {feature_range[1]})"
            )
        for i in range(len(self.max)):
            if self.max[i] <= self.min[i]:
                raise ValueError(
                    f"Mins must be inferior and different from max : Max {i} = {self.max[i]}, Min {i}={self.min[i]}"
                )

    def transform(self, x: np.array) -> np.array:
        return ((x - self.min) / (self.max - self.min)) * (
            self.feature_range[1] - self.feature_range[0]
        ) + self.feature_range[0]


class SimpleStandardizer:
    def __init__(self, clip=False, shift_mean=True, clipping_range=(-10, 10)) -> None:

        # Init internals
        self.count = 0
        self.mean = None
        self.M2 = None
        self.std = None

        # Fetch attributes
        self.shift_mean = shift_mean
        self.clip = clip
        self.clipping_range = clipping_range

    def partial_fit(self, newValue: np.array):
        # Welfor's online algorithm : https://en.m.wikipedia.org/wiki/Algorithms_for_calculating_variance
        self.count += 1
        if self.mean is None:
            self.mean = newValue
            self.std = np.zeros(len(newValue))
            self.M2 = np.zeros(len(newValue))
        delta = newValue - self.mean
        self.mean = self.mean + (delta / self.count)
        delta2 = newValue - self.mean
        self.M2 += np.multiply(delta, delta2)
        if self.count >= 2:
            self.std = np.sqrt(self.M2 / self.count)
            self.std = np.nan_to_num(self.std, nan=1)

    def transform(self, value: np.array):
        self.std[self.std == 0.0] = 1
        if self.shift_mean:
            new_value = (value - self.mean) / self.std
        else:
            new_value = value / self.std
        if self.clip:
            return np.clip(new_value, self.clipping_range[0], self.clipping_range[1])
        else:
            return new_value

    def pytorch_transform(self, value: torch.Tensor):
        self.std[self.std == 0.0] = 1
        if self.shift_mean:
            new_value = torch.div((torch.sub(value, t(self.mean))) / t(self.std))
        else:
            new_value = torch.div(value, t(self.std))
        if self.clip:
            return torch.clip(new_value, self.clipping_range[0], self.clipping_range[1])
        else:
            return new_value


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon
        self.epsilon = epsilon

    def partial_fit(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def transform(self, obs):
        self.partial_fit(obs)
        return (obs - self.mean) / np.sqrt(self.var + self.epsilon)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation:
    def __init__(
        self,
        env,
        epsilon=1e-8,
    ):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

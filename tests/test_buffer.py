import unittest
import numpy as np
from deeprlyb.utils.buffer import RolloutBuffer
import numpy as np


class TestSimpleStandardizer(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            RolloutBuffer(buffer_size=-1, gamma=0.99, n_steps=10)
        with self.assertRaises(ValueError):
            RolloutBuffer(buffer_size=10, gamma=10, n_steps=10)
        with self.assertRaises(ValueError):
            RolloutBuffer(buffer_size=10, gamma=0.99, n_steps=-1)

    def test_add(self):
        buffer = RolloutBuffer(buffer_size=5, gamma=0.99, n_steps=5)
        while not buffer.full:
            buffer.add(-1.3, False, 0.3, 0.1, 1e-3, 0.5)
        self.assertEqual(buffer._n_steps + buffer._buffer_size - 1, buffer.__len__)
        buffer = RolloutBuffer(buffer_size=5, gamma=0.99, n_steps=5)

        # check that buffer is full on done
        dones = [False, False, True, False]
        i = 0
        while not buffer.full:
            buffer.add(-1.3, dones[i], 0.3, 0.1, 1e-3, 0.5)
            i += 1
        self.assertEqual(3, buffer.__len__)

    def test_compute_n_step_return(self):
        rewards_list = [1, 2, 3, 4, 5]
        gamma = 0.99
        G = RolloutBuffer.compute_n_step_return(rewards=rewards_list, gamma=gamma)
        self.assertAlmostEqual(
            G, 1 + 0.99 * 2 + 0.99**2 * 3 + 0.99**3 * 4 + 0.99**4 * 5
        )

    def test_compute_advantages(self):
        buffer = RolloutBuffer(buffer_size=1, gamma=0.99, n_steps=1)
        while not buffer.full:
            buffer.add(np.random.randint(10), False, 1, 0.1, 1e-3, 0.5)
        buffer.update_advantages(last_val=1, fast=True)
        returns = buffer._returns
        buffer.update_advantages(last_val=1, fast=False)
        safe_returns = buffer._returns
        np.testing.assert_array_almost_equal(returns, safe_returns)

        # buffer = RolloutBuffer(buffer_size=2, gamma=0.99, n_steps=2)
        # while not buffer.full:
        #     buffer.add(1, False, 1, 0.1, 1e-3, 0.5)
        # buffer.update_advantages(last_val=1, fast=True)
        # returns = buffer._returns
        # v_n = RolloutBuffer.compute_returns(
        #     [1, 1, 1], buffer.gamma, buffer.buffer_size, buffer.n_steps
        # )
        # v_0 = RolloutBuffer.compute_returns(
        #     [1, 1, 1, 1, 1], buffer.gamma, buffer.buffer_size, buffer.n_steps
        # )
        # advantage_1 = returns[0] + (buffer.gamma ** buffer.n_steps) * v_n - v_0
        # np.testing.assert_array_almost_equal(buffer.advantages, advantage_1)

    def test_clean(self):
        buffer = RolloutBuffer(buffer_size=10, gamma=0.99, n_steps=5)
        vals = range(100)
        i = 0
        while not buffer.full:
            buffer.add(vals[i], False, 1, 0.1, 1e-3, 0.5)
            i += 1
        buffer.clean()
        self.assertEqual(buffer.n_steps - 1, buffer.__len__)
        while not buffer.full:
            buffer.add(vals[i], False, 1, 0.1, 1e-3, 0.5)
            i += 1
        self.assertEqual(buffer._n_steps + buffer._buffer_size - 1, buffer.__len__)


if __name__ == "__main__":
    unittest.main()

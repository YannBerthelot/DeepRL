import unittest
import torch
import numpy as np
from deeprlyb.network.utils import t, get_network_from_architecture
from deeprlyb.network.network import ActorCriticRecurrentNetworks


class TestNetworkUtils(unittest.TestCase):
    def test_get_network(self):
        input_shape = 16
        output_shape = 4
        # Test different architectures to assure that LSTM and multi layer LSTM works
        architectures = [
            "[256,128,64,32,16]",
            "[256,LSTM(128),64]",
            "[LSTM(128*2),12]",
        ]

        for architecture in architectures:
            input = t(np.array([np.ones(input_shape)]))
            parsed_archi = architecture[1:-1].split(",")
            network = get_network_from_architecture(
                input_shape,
                output_shape,
                parsed_archi,
                activation_function="relu",
                mode="actor",
            )

            self.assertEqual(int(len(network) / 2) - 1, len(parsed_archi))

    def test_init_and_forward(self):
        state_dim = 16
        action_dim = 4
        architecture = "[256,LSTM(128),64]"

        # Actor
        actor = True

        network = ActorCriticRecurrentNetworks(
            state_dim=state_dim,
            action_dim=action_dim,
            architecture=architecture,
            actor=actor,
        )
        hiddens = network.initialize_hidden_states()
        # simulate passes
        for i in range(3):
            input = t(np.array([np.random.randint(10, size=state_dim)]))
            output, hiddens = network.forward(input, hiddens)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(output.shape[-1] == action_dim)

        # Critic
        actor = False

        network = ActorCriticRecurrentNetworks(
            state_dim=state_dim,
            action_dim=action_dim,
            architecture=architecture,
            actor=actor,
        )
        hiddens = network.initialize_hidden_states()
        # simulate passes
        for i in range(3):
            input = t(np.array([np.random.randint(10, size=state_dim)]))
            output, hiddens = network.forward(input, hiddens)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(output.shape[-1] == 1)


if __name__ == "__main__":
    unittest.main()

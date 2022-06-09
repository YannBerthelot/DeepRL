import os
import unittest
import torch
import numpy as np
from deeprlyb.utils.normalize import SimpleStandardizer
from deeprlyb.network.utils import t


class TestSimpleStandardizer(unittest.TestCase):
    def test_init(self) -> None:
        with self.assertRaises(ValueError):
            SimpleStandardizer(clip=True, clipping_range=(5, -1e3))
        with self.assertRaises(ValueError):
            SimpleStandardizer(clip=True, clipping_range=(1, 1))

    def test_partial_fit(self) -> None:
        # Non-zero std
        standardizer = SimpleStandardizer(shift_mean=True)
        samples = np.array([np.random.randint(10, size=5) for i in range(100)])

        for sample in samples:
            standardizer.partial_fit(sample)
        # Assert that the results are close enough
        self.assertTrue(np.allclose(standardizer.mean, samples.mean(axis=0)))
        self.assertTrue(np.allclose(standardizer.std, samples.std(axis=0)))

        # Zero std
        standardizer = SimpleStandardizer(shift_mean=True)
        samples = np.array([np.zeros(4) for i in range(100)])
        for sample in samples:
            standardizer.partial_fit(sample)
        # Assert that the results are close enough
        self.assertTrue(np.allclose(standardizer.mean, samples.mean(axis=0)))
        self.assertTrue(np.allclose(standardizer.std, samples.std(axis=0)))

        with self.assertRaises(ValueError):
            standardizer = SimpleStandardizer(shift_mean=True)
            standardizer.partial_fit(np.ones(4))
            standardizer.partial_fit(np.ones(3))

    def test_transform(self) -> None:
        with self.assertRaises(TypeError):
            standardizer = SimpleStandardizer(shift_mean=True)
            standardizer.transform(np.ones(4))

        standardizer = SimpleStandardizer(shift_mean=True)
        standardizer.mean, standardizer.std = np.zeros(3), np.ones(3)
        sample = np.random.randint(10, size=3)
        result = standardizer.transform(sample)
        self.assertTrue(np.allclose(sample, result))
        self.assertIsInstance(result, np.ndarray)

    def test_pytorch_transform(self) -> None:
        with self.assertRaises(TypeError):
            standardizer = SimpleStandardizer(shift_mean=True)
            standardizer.transform(t(np.ones(4)))

        standardizer = SimpleStandardizer(shift_mean=True)
        standardizer.mean, standardizer.std = np.zeros(3), np.ones(3)
        sample = t(np.random.randint(10, size=3))
        result = standardizer.transform(sample)
        self.assertTrue(np.allclose(sample, result))
        self.assertIsInstance(result, torch.Tensor)

    def test_saving(self) -> None:
        if os.path.exists("./standardizer.pkl"):
            os.remove("./standardizer.pkl")
        standardizer = SimpleStandardizer()
        standardizer.save()
        self.assertTrue(os.path.exists("./standardizer.pkl"))
        os.remove("./standardizer.pkl")

    def test_loading(self) -> None:
        standardizer = SimpleStandardizer()
        samples = np.array([np.random.randint(10, size=5) for i in range(100)])
        for sample in samples:
            standardizer.partial_fit(sample)
        standardizer.save()
        standardizer_loaded = SimpleStandardizer()
        standardizer_loaded.load(".")
        sample = np.random.randint(10, size=5)
        self.assertTrue(
            np.array_equal(
                standardizer_loaded.transform(sample), standardizer.transform(sample)
            )
        )
        os.remove("./standardizer.pkl")


if __name__ == "__main__":
    unittest.main()

import unittest
from newton_sos import Problem
import numpy as np


class TestPyProblem(unittest.TestCase):
    def test_new(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        _ = Problem(0.1, 0.1 / 10, x_samples, f_samples)

    def test_initialize_gaussian(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        problem = Problem(0.1, 0.1 / 10, x_samples, f_samples)
        problem.initialize_native_kernel("gaussian", 1.0)

    def test_initialize_laplacian(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        problem = Problem(0.1, 0.1 / 10, x_samples, f_samples)
        problem.initialize_native_kernel("laplacian", 1.0)

    def test_initialize_unsupported_kernel(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        problem = Problem(0.1, 0.1 / 10, x_samples, f_samples)
        with self.assertRaises(RuntimeError) as context:
            problem.initialize_native_kernel("unsupported_kernel", 1.0)
        self.assertIn("Unsupported kernel type", str(context.exception))


if __name__ == "__main__":
    unittest.main()

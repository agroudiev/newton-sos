import unittest
from newton_sos import Problem
import numpy as np


class TestPyProblem(unittest.TestCase):
    def test_new(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        _ = Problem(0.1, 0.1 / 10, x_samples, f_samples)


if __name__ == "__main__":
    unittest.main()

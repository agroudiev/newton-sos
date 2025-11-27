import unittest
from newton_sos import Problem, solve, solve_parallel
import numpy as np


def create_polynomial_problem(n):
    def polynomial(x):
        return x**4 - 3 * x**3 + 2 * x**2 + x - 1

    x_samples = np.array([[-2 + i * 0.5] for i in range(n)], dtype=np.float64)
    f_samples = np.array([[polynomial(x[0])] for x in x_samples], dtype=np.float64)
    problem = Problem(0.01, 0.1 / n, x_samples, f_samples)
    problem.initialize_native_kernel("laplacian", 0.1)
    return problem


class TestPyProblem(unittest.TestCase):
    def test_new(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        problem = Problem(0.1, 0.1 / 10, x_samples, f_samples)
        self.assertTrue(problem.K is None)

    def test_initialize_gaussian(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        problem = Problem(0.1, 0.1 / 10, x_samples, f_samples)
        problem.initialize_native_kernel("gaussian", 1.0)
        self.assertTrue(problem.K is not None)

    def test_initialize_laplacian(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        problem = Problem(0.1, 0.1 / 10, x_samples, f_samples)
        problem.initialize_native_kernel("laplacian", 1.0)
        self.assertFalse(problem.K is None)

    def test_K_phi(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        problem = Problem(0.1, 0.1 / 10, x_samples, f_samples)
        problem.initialize_native_kernel("laplacian", 1.0)
        self.assertFalse(problem.K is None)
        self.assertTrue(problem.phi is None)
        problem.compute_phi()
        self.assertFalse(problem.phi is None)

    def test_initialize_unsupported_kernel(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        problem = Problem(0.1, 0.1 / 10, x_samples, f_samples)
        with self.assertRaises(RuntimeError) as context:
            problem.initialize_native_kernel("unsupported_kernel", 1.0)
        self.assertIn("Unsupported kernel type", str(context.exception))

    def test_solve_identity(self):
        x_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        f_samples = np.array([[i] for i in range(10)], dtype=np.float64)
        problem = Problem(0.1, 0.1 / 10, x_samples, f_samples)
        problem.initialize_native_kernel("gaussian", 1.0)

        solve_result = solve(
            problem, max_iter=100, verbose=False, method="partial_piv_lu"
        )
        self.assertTrue(solve_result.converged)
        self.assertLess(solve_result.iterations, 100)
        self.assertEqual(solve_result.status, "Converged in Newton decrement")

    def test_solve_polynomial(self):
        problem = create_polynomial_problem(10)

        solve_result = solve(
            problem, max_iter=100, verbose=False, method="partial_piv_lu"
        )
        self.assertTrue(solve_result.converged)
        self.assertEqual(solve_result.iterations, 10)
        self.assertEqual(solve_result.status, "Converged in Newton decrement")
        self.assertAlmostEqual(solve_result.z_hat[0, 0], 0.01939745, places=7)

    def test_solve_single_polynomial(self):
        problem = create_polynomial_problem(10)

        solve_results = solve_parallel(
            [problem], max_iter=100, verbose=False, method="partial_piv_lu"
        )
        self.assertTrue(solve_results[0].converged)
        self.assertEqual(solve_results[0].iterations, 10)
        self.assertEqual(solve_results[0].status, "Converged in Newton decrement")
        self.assertAlmostEqual(solve_results[0].z_hat[0, 0], 0.01939745, places=7)

    def test_solve_multiple_polynomial(self):
        problem = create_polynomial_problem(10)

        solve_results = solve_parallel(
            [problem for _ in range(5)],
            max_iter=100,
            verbose=False,
            method="partial_piv_lu",
        )
        for solve_result in solve_results:
            self.assertTrue(solve_result.converged)
            self.assertEqual(solve_result.iterations, 10)
            self.assertEqual(solve_result.status, "Converged in Newton decrement")
            self.assertAlmostEqual(solve_result.z_hat[0, 0], 0.01939745, places=7)


if __name__ == "__main__":
    unittest.main()

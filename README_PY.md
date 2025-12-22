# newton-sos
Damped Newton method to solve low-rank problems arising from KernelSOS and Sum-of-Squares relaxations

## Goal
The `newton_sos` package provides a Python interface to solve large-scale semidefinite programs (SDPs) that arise from Kernel Sum-of-Squares (KernelSOS) and Sum-of-Squares (SOS) relaxations. The core of the package is implemented in Rust for performance, while Python bindings are provided for ease of use.

Formally, the package aims to solve SDPs of the form:
$$\max_{c\in\mathbb{R}, B \in \mathbb{S}^n_+} c - \lambda \text{Tr}(B) + t \log \det (B) \quad \text{s.t. }\quad f_i - c = \Phi_i^T B \Phi_i, \:\:\forall i\in[\![1, N]\!]$$
where $\Phi_i$ are feature vectors derived from a chosen kernel function, and $f_i$ are given function evaluations at points $x_i$.

## Basic usage
The steps to solve an SDP using the Python bindings are similar to the Rust version. First, define a problem by creating an instance of the `Problem` class. Then, compute the kernel matrix using the `initialize_native_kernel` method with the desired kernel parameters. Finally, call the `solve` method on the `Problem` instance to solve the optimization problem:
```python
problem = Problem( ... )  # Define the problem
problem.initialize_native_kernel( ... )  # Compute the kernel matrix
result = problem.solve( ... )  # Solve the optimization problem
```

## Full example
The example below demonstrates how to use the `newton_sos` package to solve a toy polynomial optimization problem.
```python
import numpy as np
from newton_sos import Problem
from newton_sos import solve

# Number of sample points, which defines the size of the problem
n = 10


# Define the polynomial function to optimize
def polynomial(x):
    return x**4 - 3 * x**3 + 2 * x**2 + x - 1


# Generate sample points and evaluate the polynomial at those points
x_samples = np.array([[-2 + i * 0.5] for i in range(n)], dtype=np.float64)
f_samples = np.array([[polynomial(x[0])] for x in x_samples], dtype=np.float64)
# note that the data type must be float64

# Create the optimization problem
problem = Problem(0.01, 0.1 / n, x_samples, f_samples)
# Initialize the kernel matrix
problem.initialize_native_kernel("laplacian", 0.1)
# here, we chose a Laplacian kernel with bandwidth 0.1

# Run the solver
solve_result = solve(problem, max_iter=100, verbose=True, method="partial_piv_lu")

# Extract the solution
assert solve_result.converged
assert solve_result.iterations == 10
assert solve_result.status == "Converged in Newton decrement"
assert abs(solve_result.z_hat[0, 0] - 0.01939745) < 1e-7
print(f"Result: {solve_result.z_hat[0, 0]}")
```

## Citing
If you use this code in your research, please use the following citation:
```bibtex
@software{newton_sos,
  author = {Groudiev, Antoine},
  title = {{Newton-SOS}: Damped Newton method to solve low-rank problems arising from KernelSOS and Sum-of-Squares relaxations},
  url = {https://github.com/agroudiev/newton-sos},
  version = {0.2.2},
  date = {2025-12-22},
}
```
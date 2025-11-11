### This example demonstrates how to use the Newton-SOS solver
### on a toy polynomial optimization problem.

import numpy as np
from newton_sos import Problem
from newton_sos import solve

# if you encounter issues importing newton_sos, make sure the package is installed correctly

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

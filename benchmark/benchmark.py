from ksos_tools.solvers import problem
from ksos_tools.solvers import newton, external
import newton_sos

import numpy as np
import time

center = [0.0]
radius = np.pi
sampling = "linspace"
n_samples = 50
kernel = "Gauss"
sigma = 2 * np.pi / n_samples
f = np.sin
t = 1e-3 / n_samples
lambd = 1e-3
llt_method = "eigh"  # only for python-based solvers
max_iters_newton = 100

py_problem = problem.Problem(lambd=lambd, t=t)
py_problem.generate_new_samples(f, n_samples, center, radius, sampling, None)

tic = time.time()
py_problem.initialize_kernel(sigma, kernel, llt_method=llt_method)
python_init_time = time.time() - tic
x_samples, f_samples = py_problem.samples, py_problem.f_samples

# Python-based damped Newton
tic = time.time()
z_py_newton, info_here = newton.damped_newton(
    py_problem,
    iterations=max_iters_newton,
)
py_newton_time = time.time() - tic + python_init_time

# MOSEK
tic = time.time()
z_mosek, info_here = external.solve_primal(
    py_problem,
    solver="MOSEK",
    max_iters_scs=None,
)
mosek_time = time.time() - tic + python_init_time
if info_here["status"] == "infeasible":
    print("Infeasible problem detected!")

# Rust-based damped Newton
tic = time.time()
rs_problem = newton_sos.Problem(lambd, t, x_samples, f_samples.reshape(-1, 1))
rs_problem.initialize_native_kernel(kernel.lower() + "ian", sigma)
solve_result = newton_sos.solve(
    rs_problem, max_iter=max_iters_newton, verbose=False, method="partial_piv_lu"
)
rs_newton_time = time.time() - tic

print(
    "Python Newton | Time: {:.2e} | Result: {:.4e}".format(
        py_newton_time, float(z_py_newton[0])
    )
)
print(
    "Rust Newton   | Time: {:.2e} | Result: {:.4e}".format(
        rs_newton_time, float(solve_result.z_hat[0, 0])
    )
)
print(
    "MOSEK         | Time: {:.2e} | Result: {:.4e}".format(
        mosek_time, float(z_mosek[0])
    )
)

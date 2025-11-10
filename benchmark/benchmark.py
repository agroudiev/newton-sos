from ksos_tools.solvers import problem
from ksos_tools.solvers import newton, external
import newton_sos

import numpy as np
import time
import pandas as pd

data = []

center = [0.0]
radius = np.pi
sampling = "linspace"
kernel = "Gauss"
f = np.sin
llt_method = "eigh"  # only for python-based solvers
lambd = 1e-3
max_iters_newton = 100

# the first value if for warm-up and will be ignored
for i, n_samples in enumerate(
    [30, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 350, 500, 750, 800, 1000]
):
    print(f"Running benchmark for n_samples={n_samples}...")
    sigma = 2 * np.pi / n_samples
    t = 1e-3 / n_samples

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
    if n_samples <= 50:
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

    if i == 0:
        continue  # skip the first run for warm-up

    print(
        f"n_samples={n_samples}: Python Newton time = {py_newton_time:.4f}s, MOSEK time = {mosek_time if n_samples <= 50 else 'N/A'}, Rust Newton time = {rs_newton_time:.4f}s"
    )

    data.append(
        {
            "n_samples": n_samples,
            "python_newton_time": py_newton_time,
            "mosek_time": mosek_time if n_samples <= 50 else None,
            "rust_newton_time": rs_newton_time,
        }
    )

df = pd.DataFrame(data)
df.to_csv("benchmark/dfs/benchmark_results.csv", index=False)
print(df)

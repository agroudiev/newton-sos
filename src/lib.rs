//! The `newton_sos` crate defines and solves optimization problems of the form:
//! $$\max_{c\in\mathbb{R}, B \in \mathbb{S}^n_+} c - \lambda \text{Tr}(B) + t \log \det (B) \qquad \text{s.t. }\quad f_i - c = \Phi_i^T B \Phi_i, \\:\\:\forall i\in[\\![1, N]\\!]$$
//! using a damped Newton method. Such problems arise from sum-of-squares optimization,
//! especially in the Kernel Sum-of-Squares (KernelSOS) framework.
//!
//! ## Overview
//! The main components of the crate are:
//! - [`problem::Problem`]: A struct representing the optimization problem, including data and parameters.
//! - [`solver::solve`]: A function to solve the optimization problem.
//!
//! ## Feature Flags
//! - `python`: Enables Python bindings using PyO3.

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod problem;
pub mod solver;

#[cfg(feature = "python")]
mod py_problem;
#[cfg(feature = "python")]
mod py_solver;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
#[pymodule]
fn newton_sos(newton_sos: &Bound<'_, PyModule>) -> PyResult<()> {
    newton_sos.add_class::<py_problem::PyProblem>()?;

    newton_sos.add_class::<py_solver::PySolveResult>()?;
    newton_sos.add_function(wrap_pyfunction!(py_solver::py_solve, newton_sos)?)?;
    newton_sos.add_function(wrap_pyfunction!(py_solver::py_solve_parallel, newton_sos)?)?;
    Ok(())
}

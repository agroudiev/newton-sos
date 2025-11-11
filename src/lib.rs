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
    Ok(())
}

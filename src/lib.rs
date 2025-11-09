use pyo3::prelude::*;

pub mod problem;
pub mod solver;

mod py_problem;
mod py_solver;

#[cfg(test)]
mod tests;

#[pymodule]
fn newton_sos(newton_sos: &Bound<'_, PyModule>) -> PyResult<()> {
    newton_sos.add_class::<py_problem::PyProblem>()?;

    newton_sos.add_function(wrap_pyfunction!(py_solver::py_solve, newton_sos)?)?;
    Ok(())
}

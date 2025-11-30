use faer_ext::IntoNdarray;
use numpy::{PyArray2, ndarray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::py_problem::PyProblem;
use crate::solver::SystemSolveMethod;
use crate::solver::{SolveResult, solve, solve_parallel};

#[pyclass(name = "SolveResult")]
pub struct PySolveResult {
    pub inner: SolveResult,
}

#[pymethods]
impl PySolveResult {
    #[getter]
    pub fn z_hat<'py>(&self, py: Python<'py>) -> Option<Py<PyArray2<f64>>> {
        match &self.inner.z_hat {
            Some(mat) => {
                let ndarray = mat.as_ref().into_ndarray();
                let array = PyArray2::from_array(
                    py,
                    &ndarray.into_dimensionality::<ndarray::Ix2>().unwrap(),
                );
                Some(array.into())
            }
            None => None,
        }
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    pub fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    pub fn status(&self) -> String {
        self.inner.status.clone()
    }

    #[getter]
    pub fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    #[pyo3(signature = (problem))]
    #[allow(non_snake_case)]
    pub fn get_B<'py>(&self, py: Python<'py>, problem: &PyProblem) -> PyResult<Py<PyArray2<f64>>> {
        match self.inner.get_B(&problem.inner) {
            Ok(mat) => {
                let ndarray = mat.as_ref().into_ndarray();
                let array = PyArray2::from_array(
                    py,
                    &ndarray.into_dimensionality::<ndarray::Ix2>().unwrap(),
                );
                Ok(array.into())
            }
            Err(err) => Err(PyErr::new::<PyRuntimeError, _>(format!("{:#?}", err))),
        }
    }
}

#[pyfunction(name = "solve", signature = (problem, max_iter=100, verbose=false, method="partial_piv_lu"))]
pub fn py_solve(
    problem: &PyProblem,
    max_iter: usize,
    verbose: bool,
    method: &str,
) -> PyResult<PySolveResult> {
    let method = match method {
        "llt" => SystemSolveMethod::Llt,
        "partial_piv_lu" => SystemSolveMethod::PartialPivLu,
        "full_piv_lu" => SystemSolveMethod::FullPivLu,
        _ => {
            return Err(PyErr::new::<PyRuntimeError, _>(format!(
                "Unsupported method: {}. Supported methods are 'llt', 'partial_piv_lu', and 'full_piv_lu'.",
                method
            )));
        }
    };

    let result = solve(&problem.inner, max_iter, verbose, Some(method))
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:#?}", e)))?;
    Ok(PySolveResult { inner: result })
}

#[pyfunction(name = "solve_parallel", signature = (problems, max_iter=100, verbose=false, method="partial_piv_lu"))]
pub fn py_solve_parallel(
    problems: Vec<Py<PyProblem>>,
    max_iter: usize,
    verbose: bool,
    method: &str,
) -> PyResult<Vec<PySolveResult>> {
    let method = match method {
        "llt" => SystemSolveMethod::Llt,
        "partial_piv_lu" => SystemSolveMethod::PartialPivLu,
        "full_piv_lu" => SystemSolveMethod::FullPivLu,
        _ => {
            return Err(PyErr::new::<PyRuntimeError, _>(format!(
                "Unsupported method: {}. Supported methods are 'llt', 'partial_piv_lu', and 'full_piv_lu'.",
                method
            )));
        }
    };

    let problems: Vec<crate::problem::Problem> = problems
        .into_iter()
        .map(|py_problem| Python::attach(|py| py_problem.borrow(py).inner.clone()))
        .collect();
    let result = solve_parallel(&problems, max_iter, verbose, Some(method))
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:#?}", e)))?;
    Ok(result
        .into_iter()
        .map(|inner| PySolveResult { inner })
        .collect())
}

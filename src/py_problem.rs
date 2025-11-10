use crate::problem::{Kernel, Problem};
use faer_ext::IntoFaer;
use numpy::{PyReadonlyArrayDyn, ndarray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(name = "Problem")]
pub struct PyProblem {
    pub inner: Problem,
}

#[pymethods]
impl PyProblem {
    #[new]
    fn new(
        lambda: f64,
        t: f64,
        x_samples: PyReadonlyArrayDyn<f64>,
        f_samples: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<Self> {
        let x_samples_array = x_samples.as_array();
        let x_samples_mat = x_samples_array
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()
            .into_faer();
        let f_samples_array = f_samples.as_array();
        let f_samples_mat = f_samples_array
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()
            .into_faer();

        Ok(Self {
            inner: Problem::new(
                lambda,
                t,
                x_samples_mat.to_owned(),
                f_samples_mat.to_owned(),
            )
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:#?}", e)))?,
        })
    }

    #[pyo3(signature = (kernel, sigma))]
    fn initialize_native_kernel(&mut self, kernel: String, sigma: f64) -> PyResult<()> {
        let kernel = match kernel.as_str() {
            "gaussian" => Kernel::Gaussian(sigma),
            "laplacian" => Kernel::Laplacian(sigma),
            _ => {
                return Err(PyErr::new::<PyRuntimeError, _>(format!(
                    "Unsupported kernel type: {}. Supported types are 'gaussian' and 'laplacian'.",
                    kernel
                )));
            }
        };
        self.inner
            .initialize_native_kernel(kernel)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:#?}", e)))?;

        Ok(())
    }
}

use crate::problem::{Kernel, Problem};
use faer_ext::{IntoFaer, IntoNdarray};
use numpy::{PyArray2, PyReadonlyArrayDyn, ndarray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(name = "Problem")]
pub struct PyProblem {
    pub inner: Problem,
}

#[pymethods]
impl PyProblem {
    #[new]
    /// Create a new Problem instance with the given parameters.
    ///
    /// # Arguments
    /// * `lambda` - Trace penalty parameter.
    /// * `t` - Relative precision parameter.
    /// * `x_samples` - Sample points matrix of shape (n, d).
    /// * `f_samples` - Function values at the sample points of shape (n, 1).
    /// **Note**: this function does not compute the kernel matrix `K` or the features matrix `Phi`.
    /// To compute them, please call `initialize_native_kernel` and `compute_phi` respectively.
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
    /// Initialize the kernel matrix K with the specified native kernel type and parameter
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

    #[pyo3(signature = ())]
    /// Compute the features matrix Phi from the kernel matrix `K`.
    fn compute_phi(&mut self) -> PyResult<()> {
        self.inner
            .compute_phi()
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:#?}", e)))?;
        Ok(())
    }

    #[getter]
    #[allow(non_snake_case)]
    /// Get the kernel matrix `K` as a NumPy array.
    fn K(&self, py: Python) -> PyResult<Option<Py<PyArray2<f64>>>> {
        match &self.inner.K {
            Some(mat) => {
                let ndarray = mat.as_ref().into_ndarray();
                let array = PyArray2::from_array(
                    py,
                    &ndarray.into_dimensionality::<ndarray::Ix2>().unwrap(),
                );
                Ok(Some(array.into()))
            }
            None => Ok(None),
        }
    }

    #[getter]
    #[allow(non_snake_case)]
    /// Get the features matrix `Phi` as a NumPy array.
    fn phi(&self, py: Python) -> PyResult<Option<Py<PyArray2<f64>>>> {
        match &self.inner.phi {
            Some(mat) => {
                let ndarray = mat.as_ref().into_ndarray();
                let array = PyArray2::from_array(
                    py,
                    &ndarray.into_dimensionality::<ndarray::Ix2>().unwrap(),
                );
                Ok(Some(array.into()))
            }
            None => Ok(None),
        }
    }
}

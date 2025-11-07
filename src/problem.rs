//! Defines the optimization problem structure, as well as methods for computing
//! the features matrix and kernel matrix.

use faer::prelude::*;

#[allow(non_snake_case)]
#[derive(Debug, Clone)]
/// Represents an instance of the following optimization problem:
/// ```math
/// max c - lambda * trace(B) + t log det (B)
///     s.t. f_i - Phi_i^T B Phi_i >= c, i=1,...,N
///          B >= 0
/// ```
pub struct Problem {
    /// Trace penalty
    lambda: f64,
    /// Relative precision (`epsilon / n_samples`)
    t: f64,
    /// Sample points
    x_samples: Mat<f64>,
    /// Function values at the samples
    f_samples: Mat<f64>,
    /// Features matrix (columns of the Cholesky factor `R` of the kernel matrix `K`)
    phi: Option<Mat<f64>>,
    /// Kernel matrix `K`
    K: Option<Mat<f64>>,
}

impl Problem {
    /// Creates a new problem instance from samples and parameters.
    /// 
    /// The features matrix `phi` and kernel matrix `K` are not computed at this stage.
    /// 
    /// # Arguments
    /// * `lambda` - Trace penalty parameter.
    /// * `t` - Relative precision parameter.
    /// * `x_samples` - Sample points matrix.
    /// * `f_samples` - Function values at the sample points.
    pub fn new(
        lambda: f64,
        t: f64,
        x_samples: Mat<f64>,
        f_samples: Mat<f64>,
    ) -> Self {
        Self {
            lambda,
            t,
            x_samples,
            f_samples,
            phi: None,
            K: None,
        }
    }
}
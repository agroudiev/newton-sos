//! Defines the optimization problem structure, as well as methods for computing
//! the features matrix and kernel matrix.

use faer::{
    Side,
    linalg::solvers::{Llt, LltError},
    prelude::*,
};

#[derive(Debug, Clone, Copy)]
/// Enum representing the natively supported kernel types.
///
/// The following kernels are supported:
/// - Laplacian kernel with bandwidth parameter `sigma`, defined as:
///   ```math
///   k(x, y) = exp(-||x - y||_2 / sigma)
///   ```
/// - Gaussian kernel with bandwidth parameter `sigma`, defined as:
///   ```math
///   k(x, y) = exp(-||x - y||_2^2 / (2 * sigma^2))
///   ```
pub enum Kernel {
    /// Laplacian kernel with the specified bandwidth parameter.
    Laplacian(f64),
    /// Gaussian kernel with the specified bandwidth parameter.
    Gaussian(f64),
}

#[derive(Debug)]
/// Represents errors that can occur during problem setup and initialization if the `Problem` struct.
pub enum ProblemError {
    InvalidParameter(String),
    KernelAlreadyInitialized,
    FaerLltError(LltError),
}

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
    pub(crate) lambda: f64,
    /// Relative precision (`epsilon / n_samples`)
    pub(crate) t: f64,
    /// Sample points
    pub(crate) x_samples: Mat<f64>,
    /// Function values at the samples
    pub(crate) f_samples: Mat<f64>,
    /// Features matrix (columns of the Cholesky factor `R` of the kernel matrix `K`)
    pub(crate) phi: Option<Mat<f64>>,
    /// Kernel matrix `K`
    pub(crate) K: Option<Mat<f64>>,
    /// LLT decomposition of the kernel matrix `K`
    pub(crate) K_llt: Option<Llt<f64>>,
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
    ) -> Result<Self, ProblemError> {
        if x_samples.nrows() != f_samples.nrows() {
            return Err(ProblemError::InvalidParameter(format!(
                "Number of x_samples ({}) must match number of f_samples ({}).",
                x_samples.nrows(),
                f_samples.nrows()
            )));
        }
        if f_samples.ncols() != 1 {
            return Err(ProblemError::InvalidParameter(format!(
                "f_samples must be a column vector (got {} columns).",
                f_samples.ncols()
            )));
        }
        if x_samples.nrows() == 0 {
            return Err(ProblemError::InvalidParameter(format!(
                "Number of samples must be greater than zero (got {}).",
                x_samples.nrows()
            )));
        }
        if lambda < 0.0 {
            return Err(ProblemError::InvalidParameter(format!(
                "Lambda must be non-negative (got {}).",
                lambda
            )));
        }
        if t <= 0.0 {
            return Err(ProblemError::InvalidParameter(format!(
                "t must be positive (got {}).",
                t
            )));
        }

        Ok(Self {
            lambda,
            t,
            x_samples,
            f_samples,
            phi: None,
            K: None,
            K_llt: None,
        })
    }

    /// Initializes the kernel matrix `K` and features matrix `phi` using the specified native kernel.
    ///
    /// This method computes the kernel matrix based on the provided kernel type and its parameters,
    /// and then derives the features matrix from the Cholesky decomposition of the kernel matrix.
    ///
    /// # Arguments
    /// * `kernel` - The kernel type and its associated parameter.
    ///
    /// # Errors
    /// Returns `ProblemError::KernelAlreadyInitialized` if the kernel has already been initialized.
    /// Returns a faer variant of `ProblemError` if there is an error during the Cholesky decomposition.
    pub fn initialize_native_kernel(&mut self, kernel: Kernel) -> Result<(), ProblemError> {
        if self.K.is_some() || self.phi.is_some() {
            return Err(ProblemError::KernelAlreadyInitialized);
        }

        // Verify kernel parameters
        match kernel {
            Kernel::Laplacian(sigma) | Kernel::Gaussian(sigma) => {
                if sigma <= 0.0 {
                    return Err(ProblemError::InvalidParameter(format!(
                        "Kernel bandwidth parameter sigma must be positive (got {}).",
                        sigma
                    )));
                }
            }
        }

        let n_samples = self.x_samples.nrows();
        let x_samples = &self.x_samples;

        // Define the kernel function based on the selected kernel type
        let kernel_function: Box<dyn Fn(usize, usize) -> f64> = match kernel {
            Kernel::Laplacian(sigma) => Box::new(move |i: usize, j: usize| {
                let diff = x_samples.row(i) - x_samples.row(j);
                (-diff.norm_l2() / sigma).exp()
            }),
            Kernel::Gaussian(sigma) => Box::new(move |i: usize, j: usize| {
                let diff = x_samples.row(i) - x_samples.row(j);
                (-diff.norm_l2().powi(2) / (2.0 * sigma.powi(2))).exp()
            }),
        };

        // Compute the kernel matrix using the defined kernel function
        let kernel_matrix = Mat::<f64>::from_fn(n_samples, n_samples, kernel_function);
        // Compute the features matrix
        let llt = kernel_matrix
            .llt(Side::Lower)
            .map_err(ProblemError::FaerLltError)?;
        let r = llt.L();
        // TODO: implement other decompositions (LDLT, ...)

        self.K = Some(kernel_matrix);
        self.phi = Some(r.transpose().to_owned());
        self.K_llt = Some(llt);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_problem_initialization_gaussian() {
        let n = 10;
        let x_samples = Mat::<f64>::from_fn(n, 20, |i, j| (i + j) as f64);
        let f_samples = Mat::<f64>::from_fn(n, 30, |i, _| i as f64);

        let problem = Problem::new(1.0, 1.0, x_samples, f_samples);
        assert!(problem.is_ok());
        let result = problem
            .unwrap()
            .initialize_native_kernel(Kernel::Gaussian(1.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_problem_initialization_laplacian() {
        let n = 10;
        let x_samples = Mat::<f64>::from_fn(n, 20, |i, j| (i + j) as f64);
        let f_samples = Mat::<f64>::from_fn(n, 30, |i, _| i as f64);

        let problem = Problem::new(1.0, 1.0, x_samples, f_samples);
        assert!(problem.is_ok());
        let result = problem
            .unwrap()
            .initialize_native_kernel(Kernel::Laplacian(1.0));
        assert!(result.is_ok());
    }
}

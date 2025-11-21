//! Defines the main solver algorithms for the optimization problem.
//!
//! Formally, we want to solve the following optimization problem:
//! ```math
//! max c - lambda * Tr(B) + t log det (B)
//!     s.t. f_i - Phi_i^T B Phi_i >= c, i=1,...,N
//!          B >= 0
//! ```
//! The solver uses a damped Newton method to iteratively find the optimal dual variables `alpha`.

use crate::problem::Problem;
use faer::{Side, linalg::solvers::LltError, prelude::*};
use rayon::prelude::*;
use std::{f64, fmt::Debug};

#[derive(Debug)]
#[allow(non_snake_case)]
/// Result of solving the optimization problem.
pub struct SolveResult {
    /// Minimizer of the problem
    pub z_hat: Option<Mat<f64>>,
    /// Optimal value of the problem
    pub cost: Option<f64>,
    /// Dual variables at optimality
    pub alpha: Option<Mat<f64>>,
    /// Number of iterations taken to converge
    pub iterations: usize,
    /// Whether the solver converged successfully
    pub converged: bool,
    /// Status message from the solver
    pub status: String,
}

impl SolveResult {
    /// Creates a new `SolveResult` instance for a failed solve.
    fn new_failed(iterations: usize, status: String) -> Self {
        SolveResult {
            z_hat: None,
            cost: None,
            alpha: None,
            iterations,
            converged: false,
            status,
        }
    }

    /// Creates a new `SolveResult` instance for a successful solve.
    fn new_success(
        iterations: usize,
        status: String,
        z_hat: Mat<f64>,
        cost: f64,
        alpha: Mat<f64>,
    ) -> Self {
        SolveResult {
            z_hat: Some(z_hat),
            cost: Some(cost),
            alpha: Some(alpha),
            iterations,
            converged: true,
            status,
        }
    }

    #[allow(non_snake_case)]
    /// Computes the optimal B matrix
    pub fn get_B(&self, problem: &Problem) -> Result<Mat<f64>, SolveError> {
        let K = match &problem.K {
            Some(K) => K,
            None => return Err(SolveError::ProblemNotInitialized),
        };
        let mut K_tilde = K.to_owned();
        let n = K.nrows();
        let phi = problem.phi.as_ref().ok_or(SolveError::PhiNotComputed)?;

        let alpha = match &self.alpha {
            Some(alpha) => alpha,
            None => return Err(SolveError::ConvergenceFailed),
        };

        // compute K + lambda * Diag(alpha)^-1
        for i in 0..n {
            K_tilde[(i, i)] += problem.lambda / alpha[(i, 0)];
        }

        // find the least squares solution of (K + lambda * Diag(alpha)^-1) X = phi^T
        let qr = K_tilde.qr();
        let rhs = qr.solve_lstsq(&phi.transpose());

        // compute B = (t / lambda) * (I - phi * K_tilde^-1 * phi)
        Ok(problem.t / problem.lambda * (Mat::<f64>::identity(n, n) - phi * rhs))
    }
}

/// Errors that can occur during the solving process
pub enum SolveError {
    /// Called when trying to solve a problem that has not been initialized
    ProblemNotInitialized,
    /// Error during LLT decomposition
    LltError(LltError),
    /// Called when trying to retrieve B on after a non-converged solve
    ConvergenceFailed,
    /// Called when trying to retrieve B but Phi has not been computed
    PhiNotComputed,
}

impl Debug for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::ProblemNotInitialized => write!(
                f,
                "Problem not initialized: please call Problem::initialize_native_kernel before solving."
            ),
            SolveError::LltError(e) => write!(f, "LLT decomposition error: {:?}.", e),
            SolveError::ConvergenceFailed => write!(
                f,
                "Cannot retrieve B: solve did not converge, so no alpha are available."
            ),
            SolveError::PhiNotComputed => write!(
                f,
                "Cannot retrieve B: the Phi matrix was not computed. Please call Problem::compute_phi before calling get_B."
            ),
        }
    }
}

pub(crate) fn h_prime(problem: &Problem, alpha: &Mat<f64>, c: &MatRef<f64>) -> Mat<f64> {
    // FIXME: optimize
    let n = problem.f_samples.nrows();
    Mat::<f64>::from_fn(n, 1, |i, _| {
        problem.f_samples[(i, 0)] - problem.t / alpha[(i, 0)] * c[(i, i)]
    })
}

pub(crate) fn h_pprime(problem: &Problem, alpha: &Mat<f64>, c: &MatRef<f64>) -> Mat<f64> {
    // FIXME: optimize
    let n = problem.f_samples.nrows();
    Mat::<f64>::from_fn(n, n, |i, j| {
        problem.t / (alpha[(i, 0)] * alpha[(j, 0)]) * c[(i, j)] * c[(j, i)]
    })
}

/// Methods for solving the Newton system
#[derive(Debug, Clone, Copy)]
pub enum SystemSolveMethod {
    /// Cholesky decomposition (LLT), only if the matrix is not positive definite
    Llt,
    /// Partial pivoting LU decomposition (fast but less stable)
    PartialPivLu,
    /// Full pivoting LU decomposition (more stable but slower)
    FullPivLu,
}

#[allow(non_snake_case)]
/// Solves the Newton system for the given problem and dual variables `alpha`.
pub(crate) fn solve_newton_system(
    problem: &Problem,
    alpha: &Mat<f64>,
    method: &SystemSolveMethod,
) -> Result<(Mat<f64>, f64, f64), SolveError> {
    let n = problem.f_samples.nrows();
    let K = match &problem.K {
        Some(K) => K,
        None => return Err(SolveError::ProblemNotInitialized),
    };
    let mut K_tilde = K.to_owned();

    // FIXME: optimize
    for i in 0..K_tilde.nrows() {
        K_tilde[(i, i)] += problem.lambda / alpha[(i, 0)];
    }
    // TODO: find a way to avoid computing the decomposition at each iteration

    // C is the term K (K + lambd * Diag(a)^-1)^-1
    let C = match method {
        SystemSolveMethod::Llt => {
            let K_tilde_llt = K_tilde.llt(Side::Lower).map_err(SolveError::LltError)?;
            K_tilde_llt.solve(&K)
        }
        SystemSolveMethod::PartialPivLu => K_tilde.partial_piv_lu().solve(&K),
        SystemSolveMethod::FullPivLu => K_tilde.full_piv_lu().solve(&K),
    };
    let C = C.transpose();
    let H_p = h_prime(problem, alpha, &C);
    let H_pp = h_pprime(problem, alpha, &C);
    let H_pp_solver = H_pp.llt(Side::Lower).map_err(SolveError::LltError)?;

    let denominator = H_pp_solver.solve(&Mat::ones(n, 1));
    let numerator = H_pp_solver.solve(&H_p);

    let c = numerator.sum() / denominator.sum();
    let delta = numerator - c * &denominator;

    let lambda_alpha_sq = (delta.transpose() * &H_pp * &delta)[(0, 0)];

    Ok((delta, c, lambda_alpha_sq))
}

/// Solves the optimization problem using the damped Newton method.
pub fn solve(
    problem: &Problem,
    max_iter: usize,
    verbose: bool,
    method: Option<SystemSolveMethod>,
) -> Result<SolveResult, SolveError> {
    let n = problem.f_samples.nrows();
    let mut alpha = (1.0 / n as f64) * Mat::<f64>::ones(n, 1);
    let method = method.unwrap_or(SystemSolveMethod::PartialPivLu);

    if verbose {
        println!(" it |    cost   |  lambda d | step size ");
        println!("----|-----------|-----------|----------");
    }

    let mut converged = false;
    let mut status = String::new();
    let mut final_iter = None;
    let mut cost = f64::INFINITY;
    for iter in 0..max_iter {
        let (delta, new_cost, lambda_alpha_sq) = solve_newton_system(problem, &alpha, &method)?;
        cost = new_cost;
        let stepsize = 1.0 / (1.0 + (1.0 / problem.t * lambda_alpha_sq).sqrt());
        alpha -= stepsize * &delta;

        if lambda_alpha_sq < 0.0 {
            status = "Hessian is not positive definite".into();
            converged = false;
            final_iter = Some(iter);
            break;
        }

        if lambda_alpha_sq < problem.t * n as f64 {
            status = "Converged in Newton decrement".into();
            converged = true;
            final_iter = Some(iter);
            break;
        }

        if verbose {
            println!("{iter:>3} | {cost:+.4e} | {lambda_alpha_sq:+.4e} | {stepsize:+.4e}");
        }
    }

    if converged {
        let z_hat = Mat::<f64>::from_fn(problem.x_samples.ncols(), 1, |j, _| {
            (0..n)
                .map(|i| alpha[(i, 0)] * problem.x_samples[(i, j)])
                .sum()
        });

        Ok(SolveResult::new_success(
            final_iter.unwrap(),
            status,
            z_hat,
            cost,
            alpha,
        ))
    } else {
        let final_iter = match final_iter {
            Some(it) => it,
            None => {
                status = format!("Maximum iteration ({}) reached", max_iter);
                max_iter
            }
        };
        Ok(SolveResult::new_failed(final_iter, status))
    }
}

/// Solves multiple optimization problems in parallel using the damped Newton method.
pub fn solve_parallel(
    problems: &[Problem],
    max_iter: usize,
    verbose: bool,
    method: Option<SystemSolveMethod>,
) -> Result<Vec<SolveResult>, SolveError> {
    problems
        .par_iter()
        .map(|problem| solve(problem, max_iter, verbose, method))
        .collect()
}

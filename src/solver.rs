use crate::problem::Problem;
use faer::prelude::*;

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
    /// Optimal `B` matrix
    pub B: Option<Mat<f64>>,
    /// Optimal `X` matrix (also called `M`)
    pub X: Option<Mat<f64>>,
    /// Whether the solver converged successfully
    pub converged: bool,
    /// Status message from the solver
    pub status: String,
}

impl SolveResult {
    /// Creates a new `SolveResult` instance for a failed solve.
    pub fn new_failed(iterations: usize, status: String) -> Self {
        SolveResult {
            z_hat: None,
            cost: None,
            alpha: None,
            iterations,
            B: None,
            X: None,
            converged: false,
            status,
        }
    }

    /// Creates a new `SolveResult` instance for a successful solve.
    fn new_success(iterations: usize, status: String) -> Self {
        SolveResult {
            z_hat: None,
            cost: None,
            alpha: None,
            iterations,
            B: None,
            X: None,
            converged: true,
            status,
        }
    }
}

pub enum SolveError {}

fn h_prime(problem: &Problem, alpha: &Mat<f64>, c: &Mat<f64>) -> Mat<f64> {
    // TODO: optimize
    let n = problem.f_samples.nrows();
    Mat::<f64>::from_fn(n, 1, |i, _| {
        problem.f_samples[(i, 0)] - problem.t / alpha[(i, 0)] * c[(i, i)]
    })
}

fn h_pprime(problem: &Problem, alpha: &Mat<f64>, c: &Mat<f64>) -> Mat<f64> {
    // TODO: optimize
    let n = problem.f_samples.nrows();
    Mat::<f64>::from_fn(n, n, |i, j| {
        problem.t / (alpha[(i, 0)] * alpha[(j, 0)]) * c[(i, j)] * c[(j, i)]
    })
}

fn solve_newton_system(problem: &Problem, alpha: &Mat<f64>) -> (Mat<f64>, Mat<f64>, f64) {
    unimplemented!()
}

pub fn solve(problem: &Problem, max_iter: usize, verbose: bool) -> Result<SolveResult, SolveError> {
    let n = problem.f_samples.nrows();
    let mut alpha = (1.0 / n as f64) * Mat::<f64>::ones(n, 1);
    
    let mut converged = false;
    let mut status = String::new();
    for iter in 0..max_iter {
        let (delta, c, lambda_alpha_sq) = solve_newton_system(problem, &alpha);
        let stepsize = 1.0 / (1.0 + (1.0 / problem.t * lambda_alpha_sq).sqrt());
        alpha -= stepsize * &delta;

        if lambda_alpha_sq < 0.0 {
            status = "Hessian is not positive definite".into();
            converged = false;
            break;
        }

        if lambda_alpha_sq < problem.t * n as f64 {
            status = "Converged in Newton decrement".into();
            converged = true;
            break;
        }

        if verbose {
            unimplemented!("need to compute and print iteration info");
        }
    }

    if converged {
        unimplemented!("need to compute the final results");
        Ok(SolveResult::new_success(max_iter, status))
    } else {
        Ok(SolveResult::new_failed(max_iter, status))
    }
}

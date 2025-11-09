use crate::problem::Problem;
use faer::{Side, linalg::solvers::LltError, prelude::*};
use std::f64;

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
            B: None,
            X: None,
            converged: true,
            status,
        }
    }
}

#[derive(Debug)]
pub enum SolveError {
    ProblemNotInitialized,
    LltError(LltError),
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
    FullPivLu
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
        SystemSolveMethod::PartialPivLu => {
            K_tilde.partial_piv_lu().solve(&K)
        }
        SystemSolveMethod::FullPivLu => {
            K_tilde.full_piv_lu().solve(&K)
        }
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
pub fn solve(problem: &Problem, max_iter: usize, verbose: bool, method: Option<SystemSolveMethod>) -> Result<SolveResult, SolveError> {
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
            max_iter, status, z_hat, cost, alpha,
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

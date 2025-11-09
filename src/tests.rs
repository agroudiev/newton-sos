use crate::problem::Kernel;
use crate::problem::Problem;
use crate::solver::SystemSolveMethod;
use crate::solver::{h_pprime, h_prime, solve, solve_newton_system};
use approx::assert_relative_eq;
use faer::prelude::*;

fn build_polynomial_problem(n: usize) -> Problem {
    fn polynomial(x: f64) -> f64 {
        x.powi(4) - 3.0 * x.powi(3) + 2.0 * x.powi(2) + x - 1.0
    }

    let x_samples = Mat::<f64>::from_fn(n, 1, |i, _| -2.0 + i as f64 * 0.5);
    let f_samples = Mat::<f64>::from_fn(n, 1, |i, _| polynomial(x_samples[(i, 0)]));

    let lambda = 0.01;
    let t = 0.1 / n as f64;

    let mut problem = Problem::new(lambda, t, x_samples, f_samples).unwrap();
    problem
        .initialize_native_kernel(Kernel::Laplacian(0.1))
        .unwrap();
    problem
}

#[test]
fn h_prime_polynomial() {
    let n = 2;
    let problem = build_polynomial_problem(n);
    let alpha = Mat::<f64>::from_fn(n, 1, |i, _| (i + 1) as f64);
    let c = Mat::<f64>::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.5 });

    let h_p = h_prime(&problem, &alpha, &c.as_ref());
    let expected = mat![[44.95], [17.1625]];

    assert_eq!(h_p, expected);
}

#[test]
fn h_pprime_polynomial() {
    let n = 2;
    let problem = build_polynomial_problem(n);
    let alpha = Mat::<f64>::from_fn(n, 1, |i, _| (i + 1) as f64);
    let c = Mat::<f64>::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.5 });

    let h_pp = h_pprime(&problem, &alpha, &c.as_ref());
    let expected = mat![[0.05, 0.00625], [0.00625, 0.0125]];

    assert_eq!(h_pp, expected);
}

#[test]
fn polynomial_kernel_matrix() {
    let n = 2;
    let problem = build_polynomial_problem(n);
    let kernel_matrix = problem.K.as_ref().unwrap();
    let expected = mat![[1.0, 0.00673795], [0.00673795, 1.0]];

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(kernel_matrix[(i, j)], expected[(i, j)], epsilon = 1e-6);
        }
    }
}

#[test]
fn solve_newton_polynomial() {
    let n = 2;
    let problem = build_polynomial_problem(n);
    let alpha = Mat::<f64>::from_fn(n, 1, |i, _| (i + 1) as f64);

    let (delta, c, lambda_alpha_sq) =
        solve_newton_system(&problem, &alpha, &SystemSolveMethod::Llt)
            .expect("Failed to solve Newton system");
    let expected_delta = mat![[452.6398592919037], [-452.6398592919036]];
    let expected_c = 22.764461650286353;
    let expected_lambda_alpha_sq = 12577.897878226704;

    assert_eq!(delta, expected_delta);
    assert_eq!(c, expected_c);
    assert_eq!(lambda_alpha_sq, expected_lambda_alpha_sq);
}

#[test]
fn solve_polynomial() {
    let problem = build_polynomial_problem(10);

    let result = solve(&problem, 100, true, None);
    assert!(result.is_ok());

    let solution = result.unwrap();
    assert!(solution.converged);
    assert_eq!(solution.iterations, 10);
    assert_relative_eq!(solution.z_hat.unwrap()[(0, 0)], 0.01939745, epsilon = 1e-7);
}

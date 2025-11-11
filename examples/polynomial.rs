use faer::prelude::*;
use newton_sos::problem::{Kernel, Problem};
use newton_sos::solver::solve;

fn main() {
    let n = 100;

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

    let result = solve(&problem, 1000, true, None);
    assert!(result.is_ok());

    let solution = result.unwrap();
    assert!(solution.converged);

    let z_hat = solution.z_hat.unwrap();
    println!("Result: {}", z_hat[(0, 0)]);
}

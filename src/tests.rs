use faer::prelude::*;
use crate::problem::Problem;
use crate::problem::Kernel;

#[test]
fn solve_polynomial() {
    fn polynomial(x: f64) -> f64 {
        x.powi(4) - 3.0 * x.powi(3) + 2.0 * x.powi(2) + x - 1.0
    }

    let x_samples = Mat::<f64>::from_fn(10, 1, |i, _| -2.0 + i as f64 * 0.5);
    let f_samples = Mat::<f64>::from_fn(10, 1, |i, _| polynomial(x_samples[(i, 0)]));

    let lambda = 0.01;
    let t = 0.1 / 10.0;

    let mut problem = Problem::new(lambda, t, f_samples, x_samples).unwrap();
    problem.initialize_native_kernel(Kernel::Laplacian(0.1)).unwrap();
}
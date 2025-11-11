use faer::prelude::*; // for Mat
use newton_sos::problem::{Kernel, Problem};
use newton_sos::solver::solve;

fn main() {
    // Number of sample points, which defines the size of the problem
    let n = 10;

    // Define the polynomial function to optimize
    fn polynomial(x: f64) -> f64 {
        x.powi(4) - 3.0 * x.powi(3) + 2.0 * x.powi(2) + x - 1.0
    }

    // Generate sample points and evaluate the polynomial at those points
    let x_samples = Mat::<f64>::from_fn(n, 1, |i, _| -2.0 + i as f64 * 0.5);
    let f_samples = Mat::<f64>::from_fn(n, 1, |i, _| polynomial(x_samples[(i, 0)]));
    // note: x_samples can be of size n x d for d-dimensional inputs, but f_samples must be n x 1

    // Regularization parameter and "precision" parameter
    let lambda = 0.01;
    let epsilon = 0.1;
    let t = epsilon / n as f64;

    // Create the optimization problem
    let mut problem = Problem::new(lambda, t, x_samples, f_samples).unwrap();
    // Initialize the kernel matrix
    problem
        .initialize_native_kernel(Kernel::Laplacian(0.1))
        .unwrap();
    // here, we chose a Laplacian kernel with bandwidth 0.1

    // Run the solver
    let result = solve(&problem, 1000, true, None);
    assert!(result.is_ok()); // return an error if the solver fails

    // Extract the solution
    let solution = result.unwrap();
    assert!(solution.converged);

    // The solution contains the estimated minimizer z_hat
    let z_hat = solution.z_hat.unwrap();
    println!("Result: {}", z_hat[(0, 0)]);
}

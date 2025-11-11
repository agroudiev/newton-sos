# newton-sos
Damped Newton method to solve low-rank problems arising from KernelSOS and Sum-of-Squares relaxations

## Installation
This project is implemented using both Rust and Python. The Python bindings are created using [PyO3](https://pyo3.rs/), and [maturin](https://www.maturin.rs/) as the build system.

[maturin](https://www.maturin.rs/) can be installed directly using `pip`:
```bash
pip install maturin
```
To build the Rust code and install it directly as a Python package in the current environment, run:
```bash
maturin develop --release --features python
```

This is equivalent to building the wheels and installing it in the current environment:
```bash
maturin build --release --out dist --features python
pip install dist/newton_sos-*.whl
```

## Usage
### Rust
An example is provided in `examples/polynomial.rs`. To run it, use:
```bash
cargo run --example polynomial --release
```

Solving an SDP using the library can be done in three steps. First, define a problem by constructing a `Problem` struct. Then, compute the kernel matrix using the `Problem::initialize_native_kernel` method with the desired kernel parameters. Finally, call the `solve` method on the `Problem` instance to solve the optimization problem:
```rust
let problem = Problem::new( ... ); // Define the problem
problem.initialize_native_kernel( ... ); // Compute the kernel matrix
let result = solve(problem, ... ); // Solve the optimization problem
```

### Python
An example is provided in `examples/polynomial.py`. To run it, use:
```bash
python examples/polynomial.py
```
after installing the package as described above.

The steps to solve an SDP using the Python bindings are similar to the Rust version. First, define a problem by creating an instance of the `Problem` class. Then, compute the kernel matrix using the `initialize_native_kernel` method with the desired kernel parameters. Finally, call the `solve` method on the `Problem` instance to solve the optimization problem:
```python
problem = Problem( ... )  # Define the problem
problem.initialize_native_kernel( ... )  # Compute the kernel matrix
result = problem.solve( ... )  # Solve the optimization problem
```
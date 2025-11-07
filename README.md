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
maturin develop --release
```
To run unit test of all crates, run:
```bash
cargo test
```
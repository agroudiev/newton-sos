# Changelog

This file is used to track changes made to the project over time.

## [0.1.1] - 2025-12-27
### Added
- Python bindings for the following methods and attributes:
  - `problem::Problem::compute_Phi`
  - `problem::Problem::K`
  - `problem::Problem::phi`
- Added parallel computation for:
    - Kernel matrix computation in `problem::Problem::K`
    - Matrix construction in `solver::h_prime` and `solver::h_pprime`

### Fixed
- Missing documentation

## [0.1.0] - 2025-11-21
First release of the project.
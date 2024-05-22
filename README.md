# About
This repository contains the binary (`main`), input file, and processing/analysis scripts for the manuscript *Interpretable, Extensible Linear and Symbolic Regression Models for Charge Density Prediction Using a Hierarchy of Many-Body Correlation Descriptors* by Iyer, Kumar, Borda, Sadigh, Hamel, Bulatov, Lordi, and Samanta.

# Usage
This code uses Intel compilers and the Math Kernel Library (`module load intel mkl`). The binary can then be used as-is (`./main sample.in > log`).

`sample.in` is the main input file for training/prediction, consisting of
1. Gaussian widths (`alpha_C11`, `beta_C11`, ...),
2. Choice of many-body correlation descriptors (`isC11`, ...),
3. Number of Gaussians (`numG_C11`, ...) and order of polynomial terms, if any (`poly_order_C11`),
4. Spatial separation between sampled charge density grid points for training in Å (`chg_stp_jump_dist`),
5. Identifier for charge density files (`den_file_name`) and ordered range of files (`den_file_start_idx`, `den_file_end_idx`) to use for training/testing (must be named sequentially).

`python_simplex.py` performs simplex optimization to optimize the hyperparameters, i.e. Gaussian widths of even-tempered Gaussians, of a given many-body descriptor model.

`symbolic_regression/symbolic_regression.m` utilizes a genetic algorithm (using MATLAB's GPTIPS toolbox) to perform an evolutionary search through composite expressions starting from an initially well-specified feature matrix, i.e. the columns of the `Amat.txt` matrix files generated when running a particular model.

`symbolic_regression/regression_with_discovered_features.m` curates frequently-observed sub-expressions from an ensemble of symbolic regression runs and appends them to an existing feature matrix in various combinations. The predictive performance of a linear fit with these expanded matrices is then evaluated.

# License
This code is available under the terms of the MIT license.

LLNL Release No.: LLNL-CODE-863815

SPDX License Identifier: MIT

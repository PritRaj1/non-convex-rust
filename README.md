# NonConvex-RUST
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Continuous non-convex optimizers for constrained and unconstrained maximization problems. These algorithms were implemented as a side project, but they may be useful and have been open-sourced.

Sources/links to more information in the respective algorithm .md files.

## Algorithms

| Algorithm | Status |
|-----------|---------------|
| [Continuous Genetic Algorithm (CGA)](./docs/CGA.md) | [✓] |
| [Parallel Tempering (PT)](./docs/PT.md) | [✓] |
| [Tabu Search (TS)](./docs/TS.md) | [✓] |
| [Greedy Randomized Adaptive Search Procedure (GRASP)](./docs/GRASP.md) | [✓] |
| [Adam](./docs/ADAM.md) | [✓] |
| [Stochastic Gradient Ascent (SGA)](./docs/SGA.md) | [✓] |
| [Nelder-Mead](./docs/NM.md) | [✓] |
| [Limited Memory BFGS (L-BFGS)](./src/algorithms/limited_memory_bfgs/LBFGS.md) | [✘] |
| [Multi-Swarm Particle Optimization (MSPO)](./docs/MSPO.md) | [✓] |
| [Simulated Annealing (SA)](./src/algorithms/simulated_annealing/SA.md) | [✘] |
| [Differential Evolution (DE)](./docs/differential_evolution/DE.md) | [✓] |
| [Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](./docs/CMA_ES.md) | [✓] |

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
nonconvex-opt = "0.1.0"
```

## Importing

```rust
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::Config;
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction};
use nalgebra::{DVector, DMatrix}; // Works with dynamic
use nalgebra::{SVector, SMatrix}; // Works with static
```

The library works with both statically-sized and dynamically-size vectors. For dynamic examples, see [de_tests](./tests/de_tests.rs), [mspo_tests](./tests/mspo_tests.rs), [nm_tests](./tests/nm_tests.rs), [pt_tests](./tests/pt_tests.rs), or [solver_tests](./tests/solver_tests.rs). For static examples, see any other test.

## Usage

```rust
// Load config from file
let config = Config::new(include_str!("config.json")).unwrap();

// Or create config from JSON string
let config_json = r#"{
    "opt_conf": {
        "max_iter": 1000,
        "rtol": "1e-6", 
        "atol": "1e-6",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "CGA": {
            "common": {
                "num_parents": 2,
            },
            "crossover": {
                "Heuristic": {
                    "crossover_prob": 0.8
                }
            },
            "selection": {
                "Tournament": {
                    "tournament_size": 2
                }
            },
            "mutation": {
                "NonUniform": {
                    "mutation_rate": 0.23,
                    "b": 5.0
                }
            }
        }
    }
}"#;

let config = Config::new(config_json).unwrap();

let mut opt = NonConvexOpt::new(
    config,
    init_x, // Initial population - must be from nalgebra
    obj_f,  // Objective function
    Some(constraints) // Optional constraints
);

// Unconstrained optimization
let mut opt = NonConvexOpt::new(
    config,
    init_x,
    obj_f,
    None::<EmptyConstraints>
);

let result = opt.run();
```
To see the differences between setting up unconstrained and constrained problems, please refer to the [benches/](./benches) subdirectory. See the [examples/](./examples) subdirectory for more direction on using the lib.

Example configs are provided in [tests/jsons/](tests/jsons). More information on each config can be found in the respective algorithm .md files, (links above).

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b my-feature`
3. Make your changes
4. Run tests: `cargo test`
5. Run benchmarks: `cargo bench`    
    - To view the results, run:
    ```bash
    open target/criterion/report/index.html  # on macOS
    xdg-open target/criterion/report/index.html  # on Linux
    start target/criterion/report/index.html  # on Windows
    ```
6. Add sources and more information to the respective algorithm .md file - so that others can learn and share too!
7. Commit and push
8. Open a Pull Request

## License

This project is open-sourced under the [MIT License](LICENSE).
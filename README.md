# NonConvex-RUST
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">

<img src="examples/gifs/pt_kbf.gif" width="1000" alt="PT Example">

</div>

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
| [Limited Memory BFGS (L-BFGS)](./docs/LBFGS.md) | [✓] |
| [Multi-Swarm Particle Optimization (MSPO)](./docs/MSPO.md) | [✓] |
| [Simulated Annealing (SA)](./src/docs/SA.md) | [✓] |
| [Differential Evolution (DE)](./docs/DE.md) | [✓] |
| [Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](./docs/CMA_ES.md) | [✓] |
| [Tree-Structured Parzen Estimator (TPE)](./docs/TPE.md) | [✓] |
| [Cross Entropy Method (CEM)](./docs/CEM.md) | [✓] |

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
// Setup objective and constraints
#[derive(Clone)]
pub struct Kbf;

impl ObjectiveFunction<f64, U2> for Kbf
where
    DefaultAllocator: Allocator<U2>,
{
    fn f(&self, x: &SVector<f64, 2>) -> f64 {
        let sum_cos4: f64 = x.iter().map(|&xi| xi.cos().powi(4)).sum();
        let prod_cos2: f64 = x.iter().map(|&xi| xi.cos().powi(2)).product();
        let sum_ix2: f64 = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (i as f64 + 1.0) * xi * xi)
            .sum();

        (sum_cos4 - 2.0 * prod_cos2).abs() / sum_ix2.sqrt()
    }

    // Some algorithms require gradients
    fn gradient(&self, x: &SVector<f64, 2>) -> Option<SVector<f64, 2>> {
    }

    fn x_lower_bound(&self, _x: &SVector<f64, 2>) -> Option<SVector<f64, 2>> {
        Some(SVector::from_vec(vec![0.0, 0.0]))
    }

    fn x_upper_bound(&self, _x: &SVector<f64, 2>) -> Option<SVector<f64, 2>> {
        Some(SVector::from_vec(vec![10.0, 10.0]))
    }
}

#[derive(Debug, Clone)]
pub struct KbfConstraints;

impl BooleanConstraintFunction<f64, U2> for KbfConstraints
where
    DefaultAllocator: Allocator<U2>,
{
    fn g(&self, x: &SVector<f64, 2>) -> bool {
        let n = x.len();
        let product: f64 = x.iter().product();
        let sum: f64 = x.iter().sum();

        x.iter().all(|&xi| (0.0..=10.0).contains(&xi))
            && product > 0.75
            && sum < (15.0 * n as f64) / 2.0
    }
}

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
let obj_f = Kbf;
let constraints = KbfConstraints;

let mut opt = NonConvexOpt::new(
    config,
    init_x, // Initial population - must be from nalgebra
    obj_f,  // Objective function
    Some(constraints), // Optional constraints
    42 // Seed for rng
);

// Unconstrained optimization
let mut opt = NonConvexOpt::new(
    config,
    init_x,
    obj_f,
    None::<KbfConstraints>,
    42
);

let result = opt.run();
```
To see the differences between setting up unconstrained and constrained problems, please refer to the [benches/](./benches) subdirectory. See the [examples/](./examples) subdirectory for more direction on using the lib.

Example configs are provided in [tests/jsons/](tests/jsons). More information on each config can be found in the respective algorithm .md files, (links above).

## License

This project is open-sourced under the [MIT License](LICENSE).

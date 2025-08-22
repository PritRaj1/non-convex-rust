mod common;

use common::fcns::{
    QuadraticConstraints, QuadraticObjective, RosenbrockConstraints, RosenbrockObjective,
};
use nalgebra::{DMatrix, SMatrix};
use non_convex_opt::utils::config::{CEMConf, Config};
use non_convex_opt::utils::opt_prob::BooleanConstraintFunction;
use non_convex_opt::NonConvexOpt;

#[test]
fn test_basic_cem() {
    let config_json = r#"
    {
        "opt_conf": {
            "max_iter": 10,
            "rtol": "1e-6",
            "atol": "1e-6"
        },
        "alg_conf": {
            "CEM": {
                "common": {
                    "population_size": 50,
                    "elite_size": 10,
                    "initial_std": 1.0
                },
                "sampling": {},
                "adaptation": {},
                "advanced": {}
            }
        }
    }"#;

    let config: Config = serde_json::from_str(config_json).unwrap();

    let mut init_pop = DMatrix::<f64>::zeros(50, 2);
    for i in 0..50 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 2.0 - 1.0;
        }
    }

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let mut opt = NonConvexOpt::new(config, init_pop, obj_f, Some(constraints.clone()));

    for _ in 0..5 {
        opt.step();
    }

    let state = opt.alg.state();
    assert!(state.iter > 0);

    assert!(state.iter > 0);
    assert!(state.pop.nrows() > 0);
    assert!(state.pop.ncols() == 2);
    for i in 0..state.pop.nrows() {
        let x = state.pop.row(i).transpose();
        assert!(constraints.g(&x), "Solution violates constraints: {:?}", x);
    }
}

#[test]
fn test_cem_convergence() {
    let config_json = r#"
    {
        "opt_conf": {
            "max_iter": 50,
            "rtol": "1e-6",
            "atol": "1e-6"
        },
        "alg_conf": {
            "CEM": {
                "common": {
                    "population_size": 100,
                    "elite_size": 20,
                    "initial_std": 2.0
                },
                "sampling": {},
                "adaptation": {
                    "smoothing_factor": 0.8
                },
                "advanced": {
                    "use_restart_strategy": false
                }
            }
        }
    }"#;

    let config: Config = serde_json::from_str(config_json).unwrap();

    let mut init_pop = DMatrix::<f64>::zeros(100, 2);
    for i in 0..100 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 2.0 - 1.0;
        }
    }

    let obj_f = QuadraticObjective { a: 1.0, b: 1.0 };
    let constraints = QuadraticConstraints {};
    let mut opt = NonConvexOpt::new(config, init_pop, obj_f, Some(constraints));

    let mut best_fitness_history = Vec::new();

    for _ in 0..50 {
        opt.step();
        let current_best = opt.alg.state().best_f;
        best_fitness_history.push(current_best);
    }

    for i in 1..best_fitness_history.len() {
        assert!(
            best_fitness_history[i] >= best_fitness_history[i - 1],
            "Fitness should not decrease: {} -> {}",
            best_fitness_history[i - 1],
            best_fitness_history[i]
        );
    }

    let initial_fitness = best_fitness_history[0];
    let final_fitness = *best_fitness_history.last().unwrap();
    assert!(
        final_fitness > initial_fitness,
        "CEM should improve fitness over time: {} -> {}",
        initial_fitness,
        final_fitness
    );
}

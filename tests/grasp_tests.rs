mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{SMatrix, U1, U2};

use non_convex_opt::algorithms::grasp::grasp_opt::GRASP;
use non_convex_opt::utils::{
    config::GRASPConf,
    opt_prob::{BooleanConstraintFunction, OptProb, OptimizationAlgorithm},
};

#[test]
fn test_grasp_unconstrained() {
    let conf = GRASPConf {
        num_candidates: 50,
        alpha: 0.3,
        num_neighbors: 20,
        step_size: 0.1,
        perturbation_prob: 0.3,
        max_local_iter: 100,
        cache_bounds: true,
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let opt_prob = OptProb::new(
        Box::new(obj_f),
        None::<Box<dyn BooleanConstraintFunction<f64, U2>>>,
    );

    let mut grasp: GRASP<f64, U1, U2> = GRASP::new(conf, init_x.clone(), opt_prob);

    let initial_fitness = grasp.st.best_f;

    for _ in 0..10 {
        grasp.step();
    }

    assert!(grasp.st.best_f > initial_fitness);
}

#[test]
fn test_grasp_constrained() {
    let conf = GRASPConf {
        num_candidates: 50,
        alpha: 0.3,
        num_neighbors: 20,
        step_size: 0.1,
        perturbation_prob: 0.3,
        max_local_iter: 100,
        cache_bounds: true,
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut grasp: GRASP<f64, U1, U2> = GRASP::new(conf, init_x.clone(), opt_prob);

    let initial_fitness = grasp.st.best_f;

    for _ in 0..10 {
        grasp.step();
    }

    assert!(grasp.st.best_f > initial_fitness);
    assert!(grasp.st.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[test]
fn test_grasp_construction_and_local_search() {
    let conf = GRASPConf {
        num_candidates: 50,
        alpha: 0.3,
        num_neighbors: 20,
        step_size: 0.1,
        perturbation_prob: 0.3,
        max_local_iter: 100,
        cache_bounds: true,
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let grasp: GRASP<f64, U1, U2> = GRASP::new(conf, init_x.clone(), opt_prob);

    let solution = grasp.construct_solution();
    let improved = grasp.local_search(&solution);

    assert!(grasp.opt_prob.is_feasible(&solution));
    assert!(grasp.opt_prob.is_feasible(&improved));
}

#[test]
fn test_grasp_bounds() {
    let conf = GRASPConf {
        num_candidates: 50,
        alpha: 0.3,
        num_neighbors: 20,
        step_size: 0.1,
        perturbation_prob: 0.3,
        max_local_iter: 100,
        cache_bounds: true,
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut grasp: GRASP<f64, U1, U2> = GRASP::new(conf, init_x.clone(), opt_prob);

    for _ in 0..10 {
        grasp.step();
        assert!(grasp.st.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}

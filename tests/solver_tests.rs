mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::DMatrix;

use non_convex_opt::utils::{config::Config, opt_prob::ObjectiveFunction};
use non_convex_opt::NonConvexOpt;

#[test]
fn test_cga() {
    let conf = Config::new(include_str!("jsons/cga.json")).unwrap();

    let mut init_pop = DMatrix::zeros(50, 2);
    for i in 0..50 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0; // Random values in [-2, 2]
        }
    }

    let mut opt = NonConvexOpt::new(
        conf,
        init_pop.clone(),
        RosenbrockObjective { a: 1.0, b: 1.0 },
        Some(RosenbrockConstraints {}),
    );

    let initial_best_fitness: f64 = init_pop
        .row_iter()
        .map(|row| RosenbrockObjective { a: 1.0, b: 1.0 }.f(&row.transpose()))
        .fold(f64::INFINITY, |a, b| a.min(b));

    let result = opt.run();

    println!("Initial best fitness: {}", initial_best_fitness);
    println!("Best f: {}", result.best_f);

    assert!(-result.best_f.exp() < 0.01);
    assert!(result.best_f > initial_best_fitness);
}

#[test]
fn test_pt() {
    let conf = Config::new(include_str!("jsons/pt.json")).unwrap();

    let mut init_pop = DMatrix::zeros(10, 2);
    for i in 0..10 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0; // Random values in [-2, 2]
        }
    }

    let mut opt = NonConvexOpt::new(
        conf,
        init_pop.clone(),
        RosenbrockObjective { a: 1.0, b: 1.0 },
        Some(RosenbrockConstraints {}),
    );

    let initial_best_fitness: f64 = init_pop
        .row_iter()
        .map(|row| RosenbrockObjective { a: 1.0, b: 1.0 }.f(&row.transpose()))
        .fold(f64::INFINITY, |a, b| a.min(b));

    println!("Initial best fitness: {}", initial_best_fitness);

    let result = opt.run();

    println!("Best f: {}", result.best_f);

    assert!(-result.best_f.exp() < 0.01);
    assert!(result.best_f > initial_best_fitness);
}

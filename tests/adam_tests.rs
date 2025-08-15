mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{SMatrix, U1, U2};
use non_convex_opt::algorithms::adam::adam_opt::Adam;
use non_convex_opt::utils::{
    config::AdamConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_adam() {
    let conf = AdamConf {
        learning_rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.0,
        gradient_clip: 1.0,
        amsgrad: false,
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut adam = Adam::<f64, U1, U2>::new(conf, init_x.clone(), opt_prob);

    let initial_fitness = adam.st.best_f;

    for _ in 0..10 {
        adam.step();
    }

    assert!(adam.st.best_f > initial_fitness);
}

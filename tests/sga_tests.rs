mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{SMatrix, U1, U2};

use non_convex_opt::algorithms::sg_ascent::sga::SGAscent;
use non_convex_opt::utils::{
    config::SGAConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_sga() {
    let conf = SGAConf {
        learning_rate: 0.01,
        momentum: 0.9,
        gradient_clip: 1.0,
        noise_decay: 0.99,
        adaptive_noise: false,
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.9, 0.9]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sga: SGAscent<f64, U1, U2> = SGAscent::new(conf, init_x, opt_prob);
    let initial_fitness = sga.st.best_f;

    for _ in 0..50 {
        sga.step();
    }

    assert!(sga.st.best_f > initial_fitness);
    assert!(sga.st.best_x.iter().all(|&x| (0.0..=1.0).contains(&x)));
}

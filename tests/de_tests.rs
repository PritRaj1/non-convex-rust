mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::DMatrix;
use non_convex_opt::utils::opt_prob::{OptProb, OptimizationAlgorithm};

use non_convex_opt::{algorithms::differential_evolution::de::DE, utils::config::DEConf};

use non_convex_opt::utils::alg_conf::de_conf::{
    AdaptiveConf, CommonConf, DEStrategy, MutationType, StandardConf,
};

#[test]
fn test_de_basic() {
    let conf = DEConf {
        common: CommonConf {
            archive_size: 10,
            success_history_size: 50,
        },
        mutation_type: MutationType::Standard(StandardConf {
            f: 0.8,
            cr: 0.9,
            strategy: DEStrategy::Best2Bin,
        }),
    };

    let mut init_pop = DMatrix::zeros(100, 2);
    for i in 0..100 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 2.0 - 1.0
        }
    }

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut de = DE::new(conf, init_pop.clone(), opt_prob);
    let initial_fitness = de.st.best_f;

    for _ in 0..50 {
        de.step();
    }

    assert!(de.st.best_f > initial_fitness);
}

#[test]
fn test_adaptive_de() {
    let conf = DEConf {
        common: CommonConf {
            archive_size: 10,
            success_history_size: 50,
        },
        mutation_type: MutationType::Adaptive(AdaptiveConf {
            strategy: DEStrategy::Best2Bin,
            f_min: 0.4,
            f_max: 0.9,
            cr_min: 0.1,
            cr_max: 0.9,
        }),
    };

    let mut init_pop = DMatrix::zeros(100, 2);
    for i in 0..100 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 2.0 - 1.0
        }
    }

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut de = DE::new(conf, init_pop.clone(), opt_prob);
    let initial_fitness = de.st.best_f;

    // Run for more iterations to ensure improvement
    for _ in 0..50 {
        de.step();
    }

    assert!(
        de.st.best_f > initial_fitness,
        "Adaptive DE failed to improve fitness"
    );
}

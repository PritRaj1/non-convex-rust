mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{OMatrix, U2, U20};

use non_convex_opt::algorithms::cma_es::cma_es_opt::CMAES;
use non_convex_opt::utils::{
    config::CMAESConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_cmaes() {
    let conf = CMAESConf {
        num_parents: 10,
        initial_sigma: 0.3,
        use_active_cma: true,
        active_cma_ratio: 0.25,
    };

    let init_x = OMatrix::<f64, U20, U2>::from_element_generic(U20, U2, 0.5);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut cmaes: CMAES<f64, U20, U2> = CMAES::new(conf, init_x, opt_prob, 42);

    for _ in 0..20 {
        cmaes.step();
        assert!(
            cmaes.st.best_x.iter().all(|&x| (0.0..=1.0).contains(&x)),
            "Best solution violated constraints: {:?}",
            cmaes.st.best_x
        );
    }
}

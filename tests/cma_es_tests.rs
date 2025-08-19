mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{OMatrix, OVector, U2, U20};

use non_convex_opt::algorithms::cma_es::cma_es_opt::CMAES;
use non_convex_opt::utils::{
    config::CMAESConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_cmaes_initialization() {
    let conf = CMAESConf {
        num_parents: 10,
        initial_sigma: 0.5,
        use_active_cma: true,
        active_cma_ratio: 0.25,
    };

    let init_x = OMatrix::<f64, U20, U2>::from_element_generic(U20, U2, 0.5);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let cmaes = CMAES::new(conf, init_x, opt_prob);

    let pop: OMatrix<f64, U20, U2> = cmaes.st.pop;
    let fit: OVector<f64, U20> = cmaes.st.fitness;
    let constr: OVector<bool, U20> = cmaes.st.constraints;

    // Check dims - these tests are redundant with statically-sized vectors
    assert_eq!(pop.nrows(), 20);
    assert_eq!(pop.ncols(), 2);
    assert_eq!(fit.len(), 20);
    assert_eq!(constr.len(), 20);

    assert_eq!(cmaes.st.best_x.len(), 2);
    assert!(cmaes.st.best_f.is_finite());
}

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

    let mut cmaes: CMAES<f64, U20, U2> = CMAES::new(conf, init_x, opt_prob);

    for _ in 0..20 {
        cmaes.step();
        assert!(
            cmaes.st.best_x.iter().all(|&x| (0.0..=1.0).contains(&x)),
            "Best solution violated constraints: {:?}",
            cmaes.st.best_x
        );
    }
}

mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{DMatrix, DVector};

use non_convex_opt::algorithms::nelder_mead::nm::NelderMead;
use non_convex_opt::utils::{config::NelderMeadConf, opt_prob::OptProb};

#[test]
fn test_nm_new() {
    let conf = NelderMeadConf {
        alpha: 1.0,
        gamma: 2.0,
        rho: 0.5,
        sigma: 0.5,
    };

    let init_simplex = DMatrix::from_columns(&[
        DVector::from_vec(vec![1.0, 1.0]),
        DVector::from_vec(vec![2.0, 1.0]),
        DVector::from_vec(vec![1.0, 2.0]),
    ]);

    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let nm = NelderMead::new(conf, init_simplex.clone(), opt_prob);

    assert_eq!(nm.simplex.len(), 3);
    for i in 0..3 {
        assert_eq!(nm.simplex[i], init_simplex.column(i));
    }
}

#[test]
fn test_nm_centroid() {
    let conf = NelderMeadConf {
        alpha: 1.0,
        gamma: 2.0,
        rho: 0.5,
        sigma: 0.5,
    };

    let init_simplex = DMatrix::from_columns(&[
        DVector::from_vec(vec![1.0, 1.0]),
        DVector::from_vec(vec![2.0, 1.0]),
        DVector::from_vec(vec![1.0, 2.0]),
    ]);

    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let nm = NelderMead::new(conf, init_simplex, opt_prob);

    // Calculate centroid manually for comparison
    let mut expected_centroid = DVector::zeros(2);
    for (i, vertex) in nm.simplex.iter().enumerate() {
        if i != nm.simplex.len() - 1 {
            expected_centroid += vertex;
        }
    }
    expected_centroid /= (nm.simplex.len() - 1) as f64;

    // Compare with nm's calculation
    let actual_centroid = nm.centroid(nm.simplex.len() - 1);
    assert!((expected_centroid - actual_centroid).norm() < 1e-10);
}

// Optimization is difficult to test, it's easier to visualize in examples

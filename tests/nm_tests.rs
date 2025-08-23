mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{DMatrix, DVector};

use non_convex_opt::algorithms::nelder_mead::nm::NelderMead;
use non_convex_opt::utils::alg_conf::nm_conf::{
    AdvancedConf, CoefficientBounds, CommonConf, NelderMeadConf, RestartStrategy,
    StagnationDetection,
};
use non_convex_opt::utils::opt_prob::OptProb;

#[test]
fn test_nm_new() {
    let conf = NelderMeadConf {
        common: CommonConf {
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        },
        advanced: AdvancedConf {
            adaptive_parameters: true,
            restart_strategy: RestartStrategy::Stagnation {
                max_iterations: 50,
                threshold: 1e-6,
            },
            stagnation_detection: StagnationDetection {
                stagnation_window: 20,
                improvement_threshold: 1e-6,
                diversity_threshold: 1e-3,
            },
            coefficient_bounds: CoefficientBounds {
                alpha_bounds: (0.1, 3.0),
                gamma_bounds: (1.0, 5.0),
                rho_bounds: (0.1, 1.0),
                sigma_bounds: (0.1, 1.0),
            },
            adaptation_rate: 0.1,
            success_history_size: 20,
            improvement_history_size: 30,
        },
    };

    // Create simplex with 3 rows (n+1) and 2 columns (n) for 2D problem
    let init_simplex = DMatrix::from_rows(&[
        DVector::from_vec(vec![1.0, 1.0]).transpose(), // Row 1
        DVector::from_vec(vec![2.0, 1.0]).transpose(), // Row 2
        DVector::from_vec(vec![1.0, 2.0]).transpose(), // Row 3
    ]);

    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let nm = NelderMead::new(conf, init_simplex.clone(), opt_prob, 42);

    assert_eq!(nm.simplex.len(), 3);
    for i in 0..3 {
        assert_eq!(nm.simplex[i], init_simplex.row(i).transpose());
    }
}

#[test]
fn test_nm_centroid() {
    let conf = NelderMeadConf {
        common: CommonConf {
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        },
        advanced: AdvancedConf {
            adaptive_parameters: true,
            restart_strategy: RestartStrategy::Stagnation {
                max_iterations: 50,
                threshold: 1e-6,
            },
            stagnation_detection: StagnationDetection {
                stagnation_window: 20,
                improvement_threshold: 1e-6,
                diversity_threshold: 1e-3,
            },
            coefficient_bounds: CoefficientBounds {
                alpha_bounds: (0.1, 3.0),
                gamma_bounds: (1.0, 5.0),
                rho_bounds: (0.1, 1.0),
                sigma_bounds: (0.1, 1.0),
            },
            adaptation_rate: 0.1,
            success_history_size: 20,
            improvement_history_size: 30,
        },
    };

    // Create simplex with 3 rows (n+1) and 2 columns (n) for 2D problem
    let init_simplex = DMatrix::from_rows(&[
        DVector::from_vec(vec![1.0, 1.0]).transpose(), // Row 1
        DVector::from_vec(vec![2.0, 1.0]).transpose(), // Row 2
        DVector::from_vec(vec![1.0, 2.0]).transpose(), // Row 3
    ]);

    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let nm = NelderMead::new(conf, init_simplex, opt_prob, 42);

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

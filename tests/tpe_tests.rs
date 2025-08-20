mod common;

use common::fcns::{QuadraticObjective, RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{SMatrix, U2, U3, U4, U5};

use non_convex_opt::algorithms::tpe::tpe_opt::TPE;
use non_convex_opt::utils::{
    alg_conf::tpe_conf::{AdvancedConf, KernelType},
    config::TPEConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

fn create_test_conf(
    n_initial_random: usize,
    n_ei_candidates: usize,
    gamma: f64,
    kernel_type: KernelType,
    use_restart: bool,
) -> TPEConf {
    TPEConf {
        n_initial_random,
        n_ei_candidates,
        gamma,
        prior_weight: 1.0,
        kernel_type,
        max_history: 100,
        advanced: AdvancedConf {
            use_restart_strategy: use_restart,
            restart_frequency: if use_restart { 25 } else { 100 },
        },
    }
}

#[test]
fn test_basic_opt() {
    let conf = TPEConf {
        n_initial_random: 10,
        n_ei_candidates: 50,
        gamma: 0.3,
        prior_weight: 1.0,
        kernel_type: KernelType::Gaussian,
        max_history: 100,
        advanced: AdvancedConf {
            use_restart_strategy: false,
            restart_frequency: 100,
        },
    };

    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.9, 0.9].into(),
        [0.8, 0.8].into(),
        [0.7, 0.7].into(),
        [0.6, 0.6].into(),
        [0.5, 0.5].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50);
    let initial_fitness = tpe.st.best_f;

    for _ in 0..20 {
        tpe.step();
    }

    assert!(tpe.st.best_f >= initial_fitness);
    assert!(tpe.iteration > 1);
    assert!(!tpe.observations.is_empty());
}

#[test]
fn test_restart() {
    let conf = TPEConf {
        n_initial_random: 5,
        n_ei_candidates: 30,
        gamma: 0.4,
        prior_weight: 0.5,
        kernel_type: KernelType::Gaussian,
        max_history: 100,
        advanced: AdvancedConf {
            use_restart_strategy: true,
            restart_frequency: 15,
        },
    };

    let init_pop =
        SMatrix::<f64, 3, 2>::from_rows(&[[0.5, 0.5].into(), [0.4, 0.4].into(), [0.3, 0.3].into()]);

    let obj_f = QuadraticObjective { a: 2.0, b: 1.0 };
    let opt_prob = OptProb::new(Box::new(obj_f), None);

    let mut tpe: TPE<f64, U3, U2> = TPE::new(conf, init_pop, opt_prob, 50);
    let initial_fitness = tpe.st.best_f;

    for _ in 0..25 {
        tpe.step();
    }

    assert!(tpe.restart_counter > 0);
    assert!(tpe.st.best_f >= initial_fitness);
    assert!(tpe.iteration > 20);
}

#[test]
fn test_tracking() {
    let conf = TPEConf {
        n_initial_random: 4,
        n_ei_candidates: 25,
        gamma: 0.3,
        prior_weight: 1.0,
        kernel_type: KernelType::Gaussian,
        max_history: 100,
        advanced: AdvancedConf {
            use_restart_strategy: false,
            restart_frequency: 100,
        },
    };

    let init_pop = SMatrix::<f64, 4, 3>::from_rows(&[
        [0.1, 0.2, 0.3].into(),
        [0.4, 0.5, 0.6].into(),
        [0.7, 0.8, 0.9].into(),
        [0.0, 0.1, 0.2].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U4, U3> = TPE::new(conf, init_pop, opt_prob, 50);

    for _ in 0..15 {
        tpe.step();
    }

    assert!(!tpe.diversity_history.is_empty());
    assert!(tpe.diversity_history.len() <= 100);

    let (restart_count, _stagnation_count, avg_improvement, avg_diversity) =
        tpe.get_performance_stats();
    assert_eq!(restart_count, 0);
    assert!(avg_improvement >= 0.0);
    assert!(avg_diversity >= 0.0);
}

#[test]
fn test_kernels() {
    let kernel_types = vec![
        KernelType::Gaussian,
        KernelType::Epanechnikov,
        KernelType::TopHat,
        KernelType::Triangular,
    ];

    for kernel_type in kernel_types {
        let conf = TPEConf {
            n_initial_random: 3,
            n_ei_candidates: 20,
            gamma: 0.3,
            prior_weight: 1.0,
            kernel_type,
            max_history: 100,
            advanced: AdvancedConf {
                use_restart_strategy: false,
                restart_frequency: 100,
            },
        };

        let init_pop = SMatrix::<f64, 3, 2>::from_rows(&[
            [0.5, 0.5].into(),
            [0.4, 0.4].into(),
            [0.3, 0.3].into(),
        ]);

        let obj_f = QuadraticObjective { a: 1.0, b: 1.0 };
        let opt_prob = OptProb::new(Box::new(obj_f), None);

        let mut tpe: TPE<f64, U3, U2> = TPE::new(conf, init_pop, opt_prob, 50);
        let initial_fitness = tpe.st.best_f;

        for _ in 0..10 {
            tpe.step();
        }

        assert!(tpe.st.best_f >= initial_fitness);
        assert!(tpe.iteration > 1);
    }
}

#[test]
fn test_stagnation() {
    let conf = create_test_conf(2, 15, 0.5, KernelType::Gaussian, false);

    let init_pop = SMatrix::<f64, 2, 2>::from_rows(&[[0.5, 0.5].into(), [0.6, 0.6].into()]);

    let obj_f = QuadraticObjective { a: 0.0, b: 1.0 }; // Simple quadratic
    let opt_prob = OptProb::new(Box::new(obj_f), None);

    let mut tpe: TPE<f64, U2, U2> = TPE::new(conf.clone(), init_pop, opt_prob, 50);

    for _ in 0..20 {
        tpe.step();
    }

    assert!(tpe.improvement_history.len() <= 100);
    let is_stagnated = tpe.is_stagnated();
    assert!(is_stagnated == (tpe.stagnation_counter >= 50));
}

#[test]
fn test_quantile() {
    let conf = TPEConf {
        n_initial_random: 4,
        n_ei_candidates: 25,
        gamma: 0.3,
        prior_weight: 1.0,
        kernel_type: KernelType::Gaussian,
        max_history: 100,
        advanced: AdvancedConf {
            use_restart_strategy: false,
            restart_frequency: 100,
        },
    };

    let init_pop = SMatrix::<f64, 4, 2>::from_rows(&[
        [0.1, 0.1].into(),
        [0.2, 0.2].into(),
        [0.3, 0.3].into(),
        [0.4, 0.4].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U4, U2> = TPE::new(conf, init_pop, opt_prob, 50);

    for _ in 0..15 {
        tpe.step();
    }

    assert!(tpe.best_observations.len() <= tpe.observations.len());
    assert!(tpe.worst_observations.len() <= tpe.observations.len());

    if !tpe.best_observations.is_empty() && !tpe.worst_observations.is_empty() {
        let best_fitness = tpe
            .best_observations
            .iter()
            .map(|(_, f)| f)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let worst_fitness = tpe
            .worst_observations
            .iter()
            .map(|(_, f)| f)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(best_fitness >= worst_fitness);
    }
}

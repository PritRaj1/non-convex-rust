mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{SMatrix, U2, U5};

use non_convex_opt::algorithms::tpe::tpe_opt::TPE;
use non_convex_opt::utils::{
    alg_conf::tpe_conf::{
        AcquisitionConf, AcquisitionType, AdvancedConf, BandwidthConf, BandwidthMethod, KernelType,
        SamplingConf, SamplingStrategy,
    },
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
            use_adaptive_gamma: false,
            use_meta_optimization: false,
            meta_optimization_frequency: 50,
            use_early_stopping: false,
            early_stopping_patience: 50,
            use_constraint_aware: false,
        },
        bandwidth: BandwidthConf {
            method: BandwidthMethod::Silverman,
            cv_folds: 5,
            adaptation_rate: 0.1,
            min_bandwidth: 1e-6,
            max_bandwidth: 10.0,
        },
        acquisition: AcquisitionConf {
            acquisition_type: AcquisitionType::ExpectedImprovement,
            xi: 0.01,
            kappa: 2.0,
            use_entropy: false,
            entropy_weight: 0.1,
        },
        sampling: SamplingConf {
            strategy: SamplingStrategy::KDEBased,
            adaptive_noise: false,
            noise_scale: 0.1,
            use_thompson: false,
            local_search: false,
            local_search_steps: 10,
        },
    }
}

#[test]
fn test_basic() {
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
            use_adaptive_gamma: false,
            use_meta_optimization: false,
            meta_optimization_frequency: 50,
            use_early_stopping: false,
            early_stopping_patience: 50,
            use_constraint_aware: false,
        },
        bandwidth: BandwidthConf {
            method: BandwidthMethod::Silverman,
            cv_folds: 5,
            adaptation_rate: 0.1,
            min_bandwidth: 1e-6,
            max_bandwidth: 10.0,
        },
        acquisition: AcquisitionConf {
            acquisition_type: AcquisitionType::ExpectedImprovement,
            xi: 0.01,
            kappa: 2.0,
            use_entropy: false,
            entropy_weight: 0.1,
        },
        sampling: SamplingConf {
            strategy: SamplingStrategy::KDEBased,
            adaptive_noise: false,
            noise_scale: 0.1,
            use_thompson: false,
            local_search: false,
            local_search_steps: 10,
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

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50, 42);
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
    let conf = create_test_conf(5, 25, 0.3, KernelType::Gaussian, true);
    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.5, 0.5].into(),
        [0.4, 0.4].into(),
        [0.3, 0.3].into(),
        [0.2, 0.2].into(),
        [0.1, 0.1].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 10, 42);
    let initial_restart_count = tpe.restart_counter;

    for _ in 0..30 {
        tpe.step();
    }

    assert!(tpe.restart_counter >= initial_restart_count);
}

#[test]
fn test_adaptive_gamma() {
    let mut conf = create_test_conf(5, 25, 0.3, KernelType::Gaussian, false);
    conf.advanced.use_adaptive_gamma = true;

    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.5, 0.5].into(),
        [0.4, 0.4].into(),
        [0.3, 0.3].into(),
        [0.2, 0.2].into(),
        [0.1, 0.1].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50, 42);
    let initial_gamma = tpe.current_gamma;

    for _ in 0..20 {
        tpe.step();
    }

    // Gamma should have changed due to adaptation
    assert!(!tpe.gamma_history.is_empty());
    assert!(tpe.current_gamma != initial_gamma);
}

#[test]
fn test_thompson_sampling() {
    let mut conf = create_test_conf(5, 25, 0.3, KernelType::Gaussian, false);
    conf.sampling.strategy = SamplingStrategy::Thompson;
    conf.sampling.use_thompson = true;

    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.5, 0.5].into(),
        [0.4, 0.4].into(),
        [0.3, 0.3].into(),
        [0.2, 0.2].into(),
        [0.1, 0.1].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50, 42);

    for _ in 0..20 {
        tpe.step();
    }

    assert!(tpe.iteration > 1);
}

#[test]
fn test_hybrid_sampling() {
    let mut conf = create_test_conf(5, 25, 0.3, KernelType::Gaussian, false);
    conf.sampling.strategy = SamplingStrategy::Hybrid;

    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.5, 0.5].into(),
        [0.4, 0.4].into(),
        [0.3, 0.3].into(),
        [0.2, 0.2].into(),
        [0.1, 0.1].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50, 42);

    for _ in 0..20 {
        tpe.step();
    }

    assert!(tpe.iteration > 1);
}

#[test]
fn test_adaptive_bandwidth() {
    let mut conf = create_test_conf(5, 25, 0.3, KernelType::Gaussian, false);
    conf.bandwidth.method = BandwidthMethod::Adaptive;

    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.5, 0.5].into(),
        [0.4, 0.4].into(),
        [0.3, 0.3].into(),
        [0.2, 0.2].into(),
        [0.1, 0.1].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50, 42);

    for _ in 0..20 {
        tpe.step();
    }

    assert!(tpe.iteration > 1);
}

#[test]
fn test_cross_validation_bandwidth() {
    let mut conf = create_test_conf(5, 25, 0.3, KernelType::Gaussian, false);
    conf.bandwidth.method = BandwidthMethod::CrossValidation;

    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.5, 0.5].into(),
        [0.4, 0.4].into(),
        [0.3, 0.3].into(),
        [0.2, 0.2].into(),
        [0.1, 0.1].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50, 42);

    for _ in 0..20 {
        tpe.step();
    }

    assert!(tpe.iteration > 1);
}

#[test]
fn test_ucb_acquisition() {
    let mut conf = create_test_conf(5, 25, 0.3, KernelType::Gaussian, false);
    conf.acquisition.acquisition_type = AcquisitionType::UpperConfidenceBound;

    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.5, 0.5].into(),
        [0.4, 0.4].into(),
        [0.3, 0.3].into(),
        [0.2, 0.2].into(),
        [0.1, 0.1].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50, 42);

    for _ in 0..20 {
        tpe.step();
    }

    assert!(tpe.iteration > 1);
}

#[test]
fn test_entropy_search_acquisition() {
    let mut conf = create_test_conf(5, 25, 0.3, KernelType::Gaussian, false);
    conf.acquisition.acquisition_type = AcquisitionType::EntropySearch;

    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.5, 0.5].into(),
        [0.4, 0.4].into(),
        [0.3, 0.3].into(),
        [0.2, 0.2].into(),
        [0.1, 0.1].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50, 42);

    for _ in 0..20 {
        tpe.step();
    }

    assert!(tpe.iteration > 1);
}

#[test]
fn test_performance_stats() {
    let conf = create_test_conf(5, 25, 0.3, KernelType::Gaussian, false);
    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.5, 0.5].into(),
        [0.4, 0.4].into(),
        [0.3, 0.3].into(),
        [0.2, 0.2].into(),
        [0.1, 0.1].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(conf, init_pop, opt_prob, 50, 42);

    for _ in 0..10 {
        tpe.step();
    }

    let (
        _restart_count,
        _stagnation_count,
        _avg_improvement,
        _avg_diversity,
        _avg_convergence,
        avg_gamma,
        avg_noise_scale,
    ) = tpe.get_performance_stats();

    assert!(avg_gamma > 0.0);
    assert!(avg_noise_scale > 0.0);
}

mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{SMatrix, U1, U2};

use non_convex_opt::algorithms::simulated_annealing::sa::SimulatedAnnealing;
use non_convex_opt::utils::{
    config::{SAConf, RestartStrategy, StagnationDetection, CoolingScheduleType},
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_sa_basic() {
    let conf = SAConf {
        initial_temp: 100.0,
        cooling_rate: 0.95,
        step_size: 0.1,
        num_neighbors: 10,
        reheat_after: 20,
        x_min: -10.0,
        x_max: 10.0,
        advanced: non_convex_opt::utils::config::AdvancedConf {
            restart_strategy: RestartStrategy::None,
            stagnation_detection: StagnationDetection {
                stagnation_window: 20,
                improvement_threshold: 1e-6,
                diversity_threshold: 0.1,
            },
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = sa.st.best_f;

    for _ in 0..10 {
        sa.step();
    }

    assert!(sa.st.best_f > initial_fitness);
    assert!(sa.st.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[test]
fn test_sa_cooling() {
    let conf = SAConf {
        initial_temp: 100.0,
        cooling_rate: 0.95,
        step_size: 0.1,
        num_neighbors: 10,
        reheat_after: 20,
        x_min: -10.0,
        x_max: 10.0,
        advanced: non_convex_opt::utils::config::AdvancedConf {
            restart_strategy: RestartStrategy::None,
            stagnation_detection: StagnationDetection {
                stagnation_window: 20,
                improvement_threshold: 1e-6,
                diversity_threshold: 0.1,
            },
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x.clone(), opt_prob);
    let initial_temp = sa.temperature;

    for _ in 0..5 {
        sa.step();
    }
    assert!(sa.temperature < initial_temp);

    for _ in 0..21 {
        sa.step();
    }
    assert!(sa.temperature > sa.temperature * sa.conf.cooling_rate);
}

#[test]
fn test_sa_neighbor_generation() {
    let conf = SAConf {
        initial_temp: 100.0,
        cooling_rate: 0.95,
        step_size: 0.1,
        num_neighbors: 10,
        reheat_after: 20,
        x_min: -10.0,
        x_max: 10.0,
        advanced: non_convex_opt::utils::config::AdvancedConf {
            restart_strategy: RestartStrategy::None,
            stagnation_detection: StagnationDetection {
                stagnation_window: 20,
                improvement_threshold: 1e-6,
                diversity_threshold: 0.1,
            },
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x.clone(), opt_prob);

    sa.step();
    assert!(sa.st.best_x.iter().all(|&x| x >= -10.0 && x <= 10.0));
}

#[test]
fn test_sa_with_constraints() {
    let conf = SAConf {
        initial_temp: 100.0,
        cooling_rate: 0.95,
        step_size: 0.1,
        num_neighbors: 10,
        reheat_after: 20,
        x_min: -10.0,
        x_max: 10.0,
        advanced: non_convex_opt::utils::config::AdvancedConf {
            restart_strategy: RestartStrategy::None,
            stagnation_detection: StagnationDetection {
                stagnation_window: 20,
                improvement_threshold: 1e-6,
                diversity_threshold: 0.1,
            },
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x.clone(), opt_prob);

    for _ in 0..10 {
        sa.step();
        assert!(sa.st.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}

#[test]
fn test_sa_acceptance() {
    let conf = SAConf {
        initial_temp: 1.0,
        cooling_rate: 0.95,
        step_size: 0.1,
        num_neighbors: 10,
        reheat_after: 20,
        x_min: -10.0,
        x_max: 10.0,
        advanced: non_convex_opt::utils::config::AdvancedConf {
            restart_strategy: RestartStrategy::None,
            stagnation_detection: StagnationDetection {
                stagnation_window: 20,
                improvement_threshold: 1e-6,
                diversity_threshold: 0.1,
            },
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x.clone(), opt_prob);
    let initial_x = sa.st.best_x.clone();

    sa.step();

    assert_ne!(sa.st.best_x, initial_x);
}

mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{SMatrix, U1, U2};

use non_convex_opt::algorithms::simulated_annealing::sa::SimulatedAnnealing;
use non_convex_opt::utils::{
    alg_conf::sa_conf::{AdvancedConf, CoolingScheduleType, RestartStrategy},
    config::SAConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_sa_basic() {
    let conf = SAConf {
        initial_temp: 1000.0,
        cooling_rate: 0.95,
        step_size: 0.5,
        num_neighbors: 50,
        x_min: -5.0,
        x_max: 5.0,
        min_step_size_factor: 0.1,
        step_size_decay_power: 0.1,
        min_temp_factor: 0.01,
        use_adaptive_cooling: false,
        advanced: AdvancedConf {
            restart_strategy: RestartStrategy::None,
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.9, 0.9]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x, opt_prob, 50, 42);
    let initial_fitness = sa.st.best_f;

    for _ in 0..100 {
        sa.step();
    }

    assert!(sa.st.best_f > initial_fitness);
    assert!(sa.st.best_x.iter().all(|&x| (-5.0..=5.0).contains(&x)));
}

#[test]
fn test_sa_cooling() {
    let conf = SAConf {
        initial_temp: 1000.0,
        cooling_rate: 0.95,
        step_size: 0.5,
        num_neighbors: 50,
        x_min: -5.0,
        x_max: 5.0,
        min_step_size_factor: 0.1,
        step_size_decay_power: 0.1,
        min_temp_factor: 0.01,
        use_adaptive_cooling: false,
        advanced: AdvancedConf {
            restart_strategy: RestartStrategy::None,
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.9, 0.9]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x, opt_prob, 50, 42);
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
        initial_temp: 1000.0,
        cooling_rate: 0.95,
        step_size: 0.5,
        num_neighbors: 50,
        x_min: -5.0,
        x_max: 5.0,
        min_step_size_factor: 0.1,
        step_size_decay_power: 0.1,
        min_temp_factor: 0.01,
        use_adaptive_cooling: false,
        advanced: AdvancedConf {
            restart_strategy: RestartStrategy::None,
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.9, 0.9]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x, opt_prob, 50, 42);

    sa.step();
    assert!(sa.st.best_x.iter().all(|&x| (-5.0..=5.0).contains(&x)));
}

#[test]
fn test_sa_with_constraints() {
    let conf = SAConf {
        initial_temp: 1000.0,
        cooling_rate: 0.95,
        step_size: 0.5,
        num_neighbors: 50,
        x_min: -5.0,
        x_max: 5.0,
        min_step_size_factor: 0.1,
        step_size_decay_power: 0.1,
        min_temp_factor: 0.01,
        use_adaptive_cooling: false,
        advanced: AdvancedConf {
            restart_strategy: RestartStrategy::None,
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.9, 0.9]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x, opt_prob, 50, 42);

    for _ in 0..10 {
        sa.step();
        assert!(sa.st.best_x.iter().all(|&x| (-5.0..=5.0).contains(&x)));
    }
}

#[test]
fn test_sa_acceptance() {
    let conf = SAConf {
        initial_temp: 1000.0,
        cooling_rate: 0.95,
        step_size: 0.5,
        num_neighbors: 50,
        x_min: -5.0,
        x_max: 5.0,
        min_step_size_factor: 0.1,
        step_size_decay_power: 0.1,
        min_temp_factor: 0.01,
        use_adaptive_cooling: false,
        advanced: AdvancedConf {
            restart_strategy: RestartStrategy::None,
            adaptive_parameters: false,
            adaptation_rate: 0.1,
            improvement_history_size: 20,
            success_history_size: 20,
            cooling_schedule: CoolingScheduleType::Exponential,
        },
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.9, 0.9]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut sa: SimulatedAnnealing<f64, U1, U2> =
        SimulatedAnnealing::new(conf, init_x, opt_prob, 50, 42);
    let initial_x = sa.st.best_x;

    sa.step();

    assert_ne!(sa.st.best_x, initial_x);
}

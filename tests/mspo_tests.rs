mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{DMatrix, DVector};

use non_convex_opt::algorithms::multi_swarm::{
    mspo::MSPO,
    particle::Particle,
    swarm::{Swarm, SwarmConfig},
};
use non_convex_opt::utils::{
    config::{AlgConf, Config},
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_particle_update() {
    let position = DVector::from_vec(vec![0.5f64, 0.5]);
    let velocity = DVector::from_vec(vec![0.1, 0.1]);
    let mut particle = Particle::new(position, velocity, 0.0);

    let global_best = DVector::from_vec(vec![1.0, 1.0]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    particle.update_velocity_and_position(
        &global_best,
        0.729,
        2.05,
        2.05,
        &opt_prob,
        (-10.0, 10.0),
    );

    assert!(particle.position.len() == 2);
    assert!(particle.velocity.len() == 2);
}

#[test]
fn test_swarm_initialization() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    // Create initial population
    let init_pop = DMatrix::from_vec(5, 2, vec![0.5, 0.5, 0.6, 0.6, 0.4, 0.4, 0.3, 0.3, 0.7, 0.7]);

    let swarm = Swarm::new(SwarmConfig {
        num_particles: 10,
        dim: 2,
        c1: 2.05,
        c2: 2.05,
        bounds: (-10.0, 10.0),
        opt_prob: &opt_prob,
        init_pop,
        inertia_start: 0.9,
        inertia_end: 0.4,
        max_iterations: 100,
    });

    assert_eq!(swarm.particles.len(), 10);
    assert!(swarm.global_best_position.len() == 2);

    // Check particles are within bounds
    for particle in &swarm.particles {
        assert!(particle.position.iter().all(|&x| x >= -10.0 && x <= 10.0));
    }
}

#[test]
fn test_swarm_update() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let init_pop = DMatrix::from_vec(5, 2, vec![0.5, 0.5, 0.6, 0.6, 0.4, 0.4, 0.3, 0.3, 0.7, 0.7]);

    let mut swarm = Swarm::new(SwarmConfig {
        num_particles: 10,
        dim: 2,
        c1: 2.05,
        c2: 2.05,
        bounds: (-10.0, 10.0),
        opt_prob: &opt_prob,
        init_pop,
        inertia_start: 0.9,
        inertia_end: 0.4,
        max_iterations: 100,
    });

    let initial_best = swarm.global_best_fitness;
    swarm.update(&opt_prob);

    // After update, particles should still be within bounds
    for particle in &swarm.particles {
        assert!(particle.position.iter().all(|&x| x >= -10.0 && x <= 10.0));
    }

    // Best fitness should not decrease
    assert!(swarm.global_best_fitness >= initial_best);
}

#[test]
fn test_mspo() {
    let config_json = r#"{
        "opt_conf": {
            "max_iter": 100,
            "rtol": "1e-6",
            "atol": "1e-6",
            "rtol_max_iter_fraction": 1.0
        },
        "alg_conf": {
            "MSPO": {
                "num_swarms": 5,
                "swarm_size": 10,
                "c1": 2.05,
                "c2": 2.05,
                "x_min": -10.0,
                "x_max": 10.0,
                "exchange_interval": 2,
                "exchange_ratio": 0.2
            }
        }
    }"#;

    let conf = Config::new(config_json).unwrap();
    let mspo_conf = match conf.alg_conf {
        AlgConf::MSPO(mspo_conf) => mspo_conf,
        _ => panic!("Expected MSPOConf"),
    };

    // Create initial population matrix
    let init_pop = DMatrix::from_vec(
        50,
        2,
        (0..100)
            .map(|i| 0.5 + 0.1 * (i as f64))
            .collect::<Vec<f64>>(),
    );

    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut mspo = MSPO::new(mspo_conf, init_pop, opt_prob, 100);
    let initial_fitness = mspo.st.best_f;

    for _ in 0..100 {
        mspo.step();
    }

    assert!(mspo.st.best_f > initial_fitness);
}

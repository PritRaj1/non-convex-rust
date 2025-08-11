mod common;

use crate::common::fcns::{
    QuadraticConstraints, QuadraticObjective, RosenbrockConstraints, RosenbrockObjective,
};
use nalgebra::{DMatrix, DVector};
use non_convex_opt::algorithms::parallel_tempering::{
    metropolis_hastings::MetropolisHastings, pt::PT,
};
use non_convex_opt::utils::{
    alg_conf::pt_conf::{AutoConf, MALAConf, MetropolisHastingsConf, PCNConf, UpdateConf},
    config::{AlgConf, Config},
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_metropolis_hastings_accept_reject() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.1, 0.1]); // Low Rosenbrock value
    let x_new = DVector::from_vec(vec![0.9, 0.9]); // High Rosenbrock value (uphill move for maximization)

    let mut mh: MetropolisHastings<f64, nalgebra::Dyn> =
        MetropolisHastings::new(opt_prob, &UpdateConf::Auto(AutoConf {}), x_old.clone());
    let constraints_new = true;
    let t = 1.0;

    let accepted = mh.accept_reject(&x_old, &x_new, constraints_new, t);

    assert_eq!(accepted, true);
}

#[test]
fn test_metropolis_hastings_local_move() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let step_size = DMatrix::identity(2, 2);
    let mut mh = MetropolisHastings::new(opt_prob, &UpdateConf::Auto(AutoConf {}), x_old.clone());
    let x_new = mh.local_move(&x_old, &step_size, 1.0);

    assert_eq!(x_old.len(), x_new.len());
}

#[test]
fn test_metropolis_hastings_update_step_size() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let step_size = DMatrix::identity(2, 2);
    let mut mh = MetropolisHastings::new(opt_prob, &UpdateConf::Auto(AutoConf {}), x_old.clone());

    let acceptance_rate = 0.5;
    let temperature = 1.0;
    let new_step_size = mh.update_step_size(&step_size, acceptance_rate, temperature);

    assert_eq!(new_step_size.nrows(), 2);
    assert_eq!(new_step_size.ncols(), 2);
}

#[test]
fn test_pcn_local_move() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let step_size = DMatrix::identity(2, 2);

    let pcn_conf = PCNConf {
        step_size: 0.01,
        preconditioner: 1.0,
    };

    let mut mh = MetropolisHastings::new(opt_prob, &UpdateConf::PCN(pcn_conf), x_old.clone());
    let x_new = mh.local_move(&x_old, &step_size, 1.0);

    assert_eq!(x_old.len(), x_new.len());
    assert_ne!(x_old, x_new);
}

#[test]
fn test_mala_local_move() {
    // Use QuadraticObjective which has gradients available
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let step_size = DMatrix::identity(2, 2);

    let mala_conf = MALAConf { step_size: 0.01 };

    let mut mh = MetropolisHastings::new(opt_prob, &UpdateConf::MALA(mala_conf), x_old.clone());
    let x_new = mh.local_move(&x_old, &step_size, 1.0);

    assert_eq!(x_old.len(), x_new.len());
    assert_ne!(x_old, x_new);
}

#[test]
fn test_metropolis_hastings_config() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let step_size = DMatrix::identity(2, 2);

    let mh_conf = MetropolisHastingsConf {
        random_walk_step_size: 0.05,
    };

    let mut mh = MetropolisHastings::new(
        opt_prob,
        &UpdateConf::MetropolisHastings(mh_conf),
        x_old.clone(),
    );
    let x_new = mh.local_move(&x_old, &step_size, 1.0);

    assert_eq!(x_old.len(), x_new.len());
}

#[test]
fn test_pt_swap() {
    let conf = Config::new(include_str!("jsons/pt.json")).unwrap();
    let pt_conf = match conf.alg_conf {
        AlgConf::PT(pt_conf) => pt_conf,
        _ => panic!("Expected PTConf"),
    };

    let init_pop = DMatrix::from_vec(2, 2, vec![0.5, 0.5, 0.5, 0.5]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let mut pt = PT::new(pt_conf, init_pop, opt_prob, 5);

    pt.swap();

    assert_eq!(pt.population.len(), 10);
    assert_eq!(pt.population[0].nrows(), 2);
    assert_eq!(pt.population[0].ncols(), 2);
}

#[test]
fn test_pt_step() {
    let conf = Config::new(include_str!("jsons/pt.json")).unwrap();
    let pt_conf = match conf.alg_conf {
        AlgConf::PT(pt_conf) => pt_conf,
        _ => panic!("Expected PTConf"),
    };

    let init_pop = DMatrix::from_vec(2, 2, vec![0.5, 0.5, 0.5, 0.5]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let mut pt = PT::new(pt_conf, init_pop, opt_prob, 5);

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
}

#[test]
fn test_different_update_configurations() {
    let pcn_conf = Config::new(include_str!("jsons/pt_pcn.json")).unwrap();
    let pt_conf_pcn = match pcn_conf.alg_conf {
        AlgConf::PT(pt_conf) => pt_conf,
        _ => panic!("Expected PTConf"),
    };

    let init_pop = DMatrix::from_vec(2, 2, vec![0.5, 0.5, 0.5, 0.5]);
    let obj_f_pcn = RosenbrockObjective { a: 1.0, b: 1.0 }; // No gradients needed
    let constraints_pcn = RosenbrockConstraints {};
    let opt_prob_pcn = OptProb::new(Box::new(obj_f_pcn), Some(Box::new(constraints_pcn)));
    let pt_pcn = PT::new(pt_conf_pcn, init_pop.clone(), opt_prob_pcn, 5);

    let mala_conf = Config::new(include_str!("jsons/pt_mala.json")).unwrap();
    let pt_conf_mala = match mala_conf.alg_conf {
        AlgConf::PT(pt_conf) => pt_conf,
        _ => panic!("Expected PTConf"),
    };

    let obj_f_mala = QuadraticObjective { a: 1.0, b: 1.0 }; // Has gradients
    let constraints_mala = QuadraticConstraints {};
    let opt_prob_mala = OptProb::new(Box::new(obj_f_mala), Some(Box::new(constraints_mala)));
    let pt_mala = PT::new(pt_conf_mala, init_pop.clone(), opt_prob_mala, 5);

    let mh_conf = Config::new(include_str!("jsons/pt_mh.json")).unwrap();
    let pt_conf_mh = match mh_conf.alg_conf {
        AlgConf::PT(pt_conf) => pt_conf,
        _ => panic!("Expected PTConf"),
    };

    let obj_f_mh = RosenbrockObjective { a: 1.0, b: 1.0 }; // No gradients needed
    let constraints_mh = RosenbrockConstraints {};
    let opt_prob_mh = OptProb::new(Box::new(obj_f_mh), Some(Box::new(constraints_mh)));
    let pt_mh = PT::new(pt_conf_mh, init_pop, opt_prob_mh, 5);

    assert_eq!(pt_pcn.population.len(), 10);
    assert_eq!(pt_mala.population.len(), 10);
    assert_eq!(pt_mh.population.len(), 10);

    assert_eq!(pt_pcn.population[0].nrows(), 2);
    assert_eq!(pt_mala.population[0].nrows(), 2);
    assert_eq!(pt_mh.population[0].nrows(), 2);
}

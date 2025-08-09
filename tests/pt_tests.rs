mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{DMatrix, DVector};

use non_convex_opt::algorithms::parallel_tempering::{
    metropolis_hastings::MetropolisHastings, pt::PT,
};
use non_convex_opt::utils::{
    config::{AlgConf, Config},
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_metropolis_hastings_accept_reject() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let x_new = DVector::from_vec(vec![0.6, 0.6]);

    let mh: MetropolisHastings<f64, nalgebra::Dyn> =
        MetropolisHastings::new(opt_prob, 0.1, 0.1, 2.1, x_old.clone());
    let constraints_new = true;
    let t = 1.0;
    let t_swap = 2.0;

    let accepted = mh.accept_reject(&x_old, &x_new, constraints_new, t, t_swap);

    assert_eq!(accepted, false);
}

#[test]
fn test_metropolis_hastings_local_move() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let step_size = DMatrix::identity(2, 2);
    let mh = MetropolisHastings::new(opt_prob, 0.1, 0.1, 2.1, x_old.clone());
    let x_new = mh.local_move(&x_old, &step_size, 1.0);

    assert_eq!(x_old.len(), x_new.len());
}

#[test]
fn test_metropolis_hastings_update_step_size() {
    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let x_new = DVector::from_vec(vec![0.6, 0.6]);
    let mut step_size = DMatrix::identity(2, 2);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let mh = MetropolisHastings::new(opt_prob, 0.1, 0.1, 2.1, x_old.clone());
    step_size = mh.update_step_size(&mut step_size, &x_old, &x_new);

    assert_eq!(step_size.nrows(), 2);
    assert_eq!(step_size.ncols(), 2);
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

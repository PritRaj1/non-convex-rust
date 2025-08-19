mod common;

use crate::common::fcns::{
    QuadraticConstraints, QuadraticObjective, RosenbrockConstraints, RosenbrockObjective,
};
use nalgebra::{DMatrix, DVector};
use non_convex_opt::algorithms::parallel_tempering::{
    metropolis_hastings::MetropolisHastings,
    preconditioners::{
        AdaptiveCovariance, FitnessWeightedCovariance, Preconditioner, SampleCovariance,
        ShrinkageCovariance,
    },
    pt::PT,
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

    let x_old = DVector::from_vec(vec![0.1, 0.1]); // High Rosenbrock value
    let x_new = DVector::from_vec(vec![0.95, 0.95]); // Lower Rosenbrock value, but closer to optimum

    let mh: MetropolisHastings<f64, nalgebra::Dyn> =
        MetropolisHastings::new(opt_prob, &UpdateConf::Auto(AutoConf {}), x_old.clone());
    let constraints_new = true;

    let x_better = DVector::from_vec(vec![0.95, 0.9025]); // Closer to Rosenbrock optimum [1,1]
    let accepted_uphill = mh.accept_reject(&x_old, &x_better, constraints_new, 0.5);
    assert!(accepted_uphill);

    let accepted_constrained = mh.accept_reject(&x_old, &x_new, false, 0.5);
    assert!(!accepted_constrained);
}

#[test]
fn test_metropolis_hastings_local_move() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let step_size = DMatrix::identity(2, 2);
    let mh = MetropolisHastings::new(opt_prob, &UpdateConf::Auto(AutoConf {}), x_old.clone());
    let x_new = mh.local_move(&x_old, &step_size, 1.0);

    assert_eq!(x_old.len(), x_new.len());
}

#[test]
fn test_metropolis_hastings_update_step_size_parks() {
    let step_size = DMatrix::identity(2, 2);
    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let x_new = DVector::from_vec(vec![0.6, 0.4]);
    let alpha = 0.1; // Learning rate
    let omega = 2.0; // Scaling factor

    let new_step_size =
        MetropolisHastings::update_step_size_parks(&step_size, &x_old, &x_new, alpha, omega);

    assert_eq!(new_step_size.nrows(), 2);
    assert_eq!(new_step_size.ncols(), 2);
    assert_ne!(new_step_size, step_size);
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

    let mh = MetropolisHastings::new(opt_prob, &UpdateConf::PCN(pcn_conf), x_old.clone());
    let x_new = mh.local_move(&x_old, &step_size, 1.0);

    assert_eq!(x_old.len(), x_new.len());
    assert_ne!(x_old, x_new);
}

#[test]
fn test_mala_local_move() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let step_size = DMatrix::identity(2, 2);

    let mala_conf = MALAConf {
        step_size: 0.01,
        use_preconditioning: false,
    };

    let mh = MetropolisHastings::new(opt_prob, &UpdateConf::MALA(mala_conf), x_old.clone());
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

    let mh = MetropolisHastings::new(
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

    assert_eq!(pt.get_num_replicas(), 10);
    assert_eq!(pt.get_replica_population(0).unwrap().nrows(), 2);
    assert_eq!(pt.get_replica_population(0).unwrap().ncols(), 2);
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

    assert_eq!(pt_pcn.get_num_replicas(), 10);
    assert_eq!(pt_mala.get_num_replicas(), 10);
    assert_eq!(pt_mh.get_num_replicas(), 10);

    assert_eq!(pt_pcn.get_replica_population(0).unwrap().nrows(), 2);
    assert_eq!(pt_mala.get_replica_population(0).unwrap().nrows(), 2);
    assert_eq!(pt_mh.get_replica_population(0).unwrap().nrows(), 2);
}

fn create_test_pt_pcn() -> PT<f64, nalgebra::Dyn, nalgebra::Dyn> {
    let conf = Config::new(include_str!("jsons/pt_pcn.json")).unwrap();
    let pt_conf = match conf.alg_conf {
        AlgConf::PT(pt_conf) => pt_conf,
        _ => panic!("Expected PTConf"),
    };

    // Need fairly large pop for covariance to change
    let init_pop = DMatrix::from_vec(10, 2, vec![
        0.90, 0.90,  // 0.81
        0.95, 0.80,  // 0.76
        0.88, 0.86,  // 0.7568
        0.92, 0.85,  // 0.782
        0.89, 0.95,  // 0.8455
        0.97, 0.78,  // 0.7566
        0.90, 0.86,  // 0.774
        0.88, 0.90,  // 0.792
        0.93, 0.90,  // 0.837
        0.91, 0.91,  // 0.8281
    ]);
    let obj_f = QuadraticObjective { a: 1.0, b: 1.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    PT::new(pt_conf, init_pop, opt_prob, 20)
}

#[test]
fn test_sample_covariance_preconditioner() {
    let mut pt = create_test_pt_pcn();
    let initial_best = pt.st.best_f;

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(SampleCovariance::new(0.001));
    pt.set_preconditioner(preconditioner);

    assert_eq!(pt.covariance_matrices.len(), pt.get_num_replicas());

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f >= initial_best || (pt.st.best_f - initial_best).abs() < 1e-10);

    for cov in &pt.covariance_matrices {
        for i in 0..cov.nrows() {
            assert!(cov[(i, i)] > 0.0, "Diagonal element should be positive");
        }

        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < 1e-10,
                    "Matrix should be symmetric"
                );
            }
        }
    }
}

#[test]
fn test_fitness_weighted_covariance_preconditioner() {
    let mut pt = create_test_pt_pcn();
    let initial_best = pt.st.best_f;

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(FitnessWeightedCovariance::new(0.01, 0.5));
    pt.set_preconditioner(preconditioner);

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f >= initial_best || (pt.st.best_f - initial_best).abs() < 1e-10);

    for cov in &pt.covariance_matrices {
        assert!(
            cov.determinant() > 0.0,
            "Covariance matrix should be positive definite"
        );

        for i in 0..cov.nrows() {
            assert!(
                cov[(i, i)] >= 0.009,
                "Diagonal should include regularization"
            );
        }
    }
}

#[test]
fn test_adaptive_covariance_preconditioner() {
    let mut pt = create_test_pt_pcn();
    let initial_best = pt.st.best_f;

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(AdaptiveCovariance::new(0.01, 0.1, 0.234));
    pt.set_preconditioner(preconditioner);

    let _initial_trace: f64 = pt.covariance_matrices[0].trace();

    for _ in 0..10 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f >= initial_best || (pt.st.best_f - initial_best).abs() < 1e-10);

    for cov in &pt.covariance_matrices {
        assert!(
            cov.determinant() > 0.0,
            "Covariance matrix should be positive definite"
        );

        let condition_number = {
            let eigenvalues = cov.symmetric_eigenvalues();
            let max_eig = eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_eig = eigenvalues
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b.max(1e-12)));
            max_eig / min_eig
        };
        assert!(
            condition_number < 1e6,
            "Condition number should be reasonable"
        );
    }
}

#[test]
fn test_shrinkage_covariance_preconditioner() {
    let mut pt = create_test_pt_pcn();
    let initial_best = pt.st.best_f;

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(ShrinkageCovariance::new(0.3));
    pt.set_preconditioner(preconditioner);

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f >= initial_best || (pt.st.best_f - initial_best).abs() < 1e-10);

    for cov in &pt.covariance_matrices {
        assert!(
            cov.determinant() > 0.0,
            "Covariance matrix should be positive definite"
        );

        let trace = cov.trace();
        let avg_diagonal = trace / cov.nrows() as f64;

        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                if i != j {
                    assert!(
                        cov[(i, j)].abs() <= avg_diagonal,
                        "Off-diagonal elements should be shrunk"
                    );
                }
            }
        }
    }
}

#[test]
fn test_preconditioner_covariance_update() {
    let mut pt = create_test_pt_pcn();

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(SampleCovariance::new(0.0001));
    pt.set_preconditioner(preconditioner);

    let initial_covariances: Vec<_> = pt.covariance_matrices.clone();

    for _ in 0..25 {
        pt.step();
    }

    let updated = pt
        .covariance_matrices
        .iter()
        .zip(initial_covariances.iter())
        .any(|(new_cov, old_cov)| {
            for i in 0..new_cov.nrows() {
                for j in 0..new_cov.ncols() {
                    if (new_cov[(i, j)] - old_cov[(i, j)]).abs() > 1e-12 {
                        return true;
                    }
                }
            }
            false
        });

    assert!(
        updated,
        "Covariance matrices should be updated after sufficient iterations"
    );
}

#[test]
fn test_pcn_variance_parameter_decrease() {
    let mut pt = create_test_pt_pcn();

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(SampleCovariance::new(0.01));
    pt.set_preconditioner(preconditioner);

    let initial_best = pt.st.best_f;

    for _ in 0..15 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f > -1e6, "Algorithm should not diverge");

    assert!(
        pt.st.best_f >= initial_best - 1.0,
        "Algorithm should not deteriorate significantly"
    );
}

#[test]
fn test_preconditioner_with_infeasible_individuals() {
    let mut pt = create_test_pt_pcn();

    for replica_idx in 0..pt.get_num_replicas() {
        pt.replicas[replica_idx].population[(0, 0)] = -1.0; // Outside QuadraticConstraints bounds
        pt.replicas[replica_idx].population[(0, 1)] = 2.0; // Outside QuadraticConstraints bounds
        pt.replicas[replica_idx].constraints[0] = false;
    }

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(SampleCovariance::new(0.01));
    pt.set_preconditioner(preconditioner);

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());

    for cov in &pt.covariance_matrices {
        assert!(
            cov.determinant() > 0.0,
            "Should compute valid covariance despite infeasible individuals"
        );
    }
}

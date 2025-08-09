mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{SMatrix, U1, U2};

use non_convex_opt::algorithms::limited_memory_bfgs::lbfgs::LBFGS;
use non_convex_opt::utils::{
    alg_conf::lbfgs_conf::{
        BacktrackingConf, CommonConf, GoldenSectionConf, HagerZhangConf, LBFGSConf, LineSearchConf,
        MoreThuenteConf, StrongWolfeConf,
    },
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_lbfgs() {
    let conf = LBFGSConf {
        common: CommonConf { memory_size: 10 },
        line_search: LineSearchConf::HagerZhang(HagerZhangConf {
            c1: 1e-4,
            c2: 0.9,
            theta: 0.5,
            gamma: 0.5,
            max_iters: 100,
        }),
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut lbfgs: LBFGS<f64, U1, U2> = LBFGS::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = lbfgs.st.best_f;

    for _ in 0..10 {
        lbfgs.step();
    }

    assert!(lbfgs.st.best_f > initial_fitness);
    assert!(lbfgs.st.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[test]
fn test_backtracking_line_search() {
    let conf = LBFGSConf {
        common: CommonConf { memory_size: 10 },
        line_search: LineSearchConf::Backtracking(BacktrackingConf { c1: 1e-4, rho: 0.5 }),
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut lbfgs: LBFGS<f64, U1, U2> = LBFGS::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = lbfgs.st.best_f;

    // Run a few iterations
    for _ in 0..5 {
        lbfgs.step();
    }

    assert!(lbfgs.st.best_f > initial_fitness);
}

#[test]
fn test_strong_wolfe_line_search() {
    let conf = LBFGSConf {
        common: CommonConf { memory_size: 10 },
        line_search: LineSearchConf::StrongWolfe(StrongWolfeConf {
            c1: 1e-4,
            c2: 0.9,
            max_iters: 100,
        }),
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut lbfgs: LBFGS<f64, U1, U2> = LBFGS::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = lbfgs.st.best_f;

    for _ in 0..5 {
        lbfgs.step();
    }

    assert!(lbfgs.st.best_f > initial_fitness);
}

#[test]
fn test_hager_zhang_line_search() {
    let conf = LBFGSConf {
        common: CommonConf { memory_size: 10 },
        line_search: LineSearchConf::HagerZhang(HagerZhangConf {
            c1: 1e-4,
            c2: 0.9,
            theta: 0.5,
            gamma: 0.5,
            max_iters: 100,
        }),
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut lbfgs: LBFGS<f64, U1, U2> = LBFGS::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = lbfgs.st.best_f;

    for _ in 0..5 {
        lbfgs.step();
    }

    assert!(lbfgs.st.best_f > initial_fitness);
}

#[test]
fn test_more_thuente_line_search() {
    let conf = LBFGSConf {
        common: CommonConf { memory_size: 10 },
        line_search: LineSearchConf::MoreThuente(MoreThuenteConf {
            ftol: 1e-4,
            gtol: 0.9,
            max_iters: 100,
        }),
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut lbfgs: LBFGS<f64, U1, U2> = LBFGS::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = lbfgs.st.best_f;

    for _ in 0..5 {
        lbfgs.step();
    }

    assert!(lbfgs.st.best_f > initial_fitness);
}

#[test]
fn test_golden_section_line_search() {
    let conf = LBFGSConf {
        common: CommonConf { memory_size: 10 },
        line_search: LineSearchConf::GoldenSection(GoldenSectionConf {
            tol: 1e-6,
            max_iters: 100,
            bracket_factor: 2.0,
        }),
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut lbfgs: LBFGS<f64, U1, U2> = LBFGS::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = lbfgs.st.best_f;

    for _ in 0..5 {
        lbfgs.step();
    }

    assert!(lbfgs.st.best_f > initial_fitness);
}

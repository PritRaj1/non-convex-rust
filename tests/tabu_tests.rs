mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{SMatrix, U1, U2};

use non_convex_opt::algorithms::tabu_search::tabu::TabuSearch;
use non_convex_opt::utils::{
    config::{AlgConf, Config},
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_standard_tabu() {
    let conf_json = r#"{
        "opt_conf": {
            "max_iter": 100,
            "rtol": "1e-6",
            "atol": "1e-6"
        },
        "alg_conf": {
            "TS": {
                "common": {
                    "tabu_list_size": 20,
                    "num_neighbors": 50,
                    "step_size": 0.1,
                    "perturbation_prob": 0.3,
                    "tabu_threshold": 1e-6
                },
                "list_type": {
                    "Standard": {}
                }
            }
        }
    }"#;

    let conf = Config::new(conf_json).unwrap();
    let tabu_conf = match conf.alg_conf {
        AlgConf::TS(tabu_conf) => tabu_conf,
        _ => panic!("Expected TabuConf"),
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tabu: TabuSearch<f64, U1, U2> = TabuSearch::new(tabu_conf, init_x.clone(), opt_prob);
    let initial_fitness = tabu.st.best_f;

    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.st.best_f > initial_fitness);
    assert!(tabu.st.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[test]
fn test_reactive_tabu() {
    let conf = Config::new(include_str!("jsons/tabu.json")).unwrap();

    let tabu_conf = match conf.alg_conf {
        AlgConf::TS(tabu_conf) => tabu_conf,
        _ => panic!("Expected TabuConf"),
    };

    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tabu: TabuSearch<f64, U1, U2> = TabuSearch::new(tabu_conf, init_x.clone(), opt_prob);
    let initial_fitness = tabu.st.best_f;

    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.st.best_f > initial_fitness);
    assert!(tabu.st.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

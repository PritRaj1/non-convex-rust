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
                },
                "advanced": {
                    "adaptive_parameters": true,
                    "aspiration_criteria": true,
                    "neighborhood_strategy": {
                        "Uniform": {
                            "step_size": 0.1,
                            "prob": 0.3
                        }
                    },
                    "restart_strategy": {
                        "Stagnation": {
                            "max_iterations": 50,
                            "threshold": 1e-6
                        }
                    },
                    "intensification_cycles": 5,
                    "diversification_threshold": 0.1,
                    "success_history_size": 20,
                    "adaptation_rate": 0.1
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

    let mut tabu: TabuSearch<f64, U1, U2> = TabuSearch::new(tabu_conf, init_x, opt_prob);
    let initial_fitness = tabu.st.best_f;

    for _ in 0..30 {
        tabu.step();
    }

    assert!(tabu.st.best_f > initial_fitness);
    assert!(tabu.st.best_x.iter().all(|&x| (0.0..=1.0).contains(&x)));
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

    let mut tabu: TabuSearch<f64, U1, U2> = TabuSearch::new(tabu_conf, init_x, opt_prob);
    let initial_fitness = tabu.st.best_f;

    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.st.best_f > initial_fitness);
    assert!(tabu.st.best_x.iter().all(|&x| (-5.0..=5.0).contains(&x)));
}

#[test]
fn test_frequency_based_tabu() {
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
                    "FrequencyBased": {
                        "frequency_threshold": 3,
                        "max_frequency": 10
                    }
                },
                "advanced": {
                    "adaptive_parameters": true,
                    "aspiration_criteria": true,
                    "neighborhood_strategy": {
                        "Gaussian": {
                            "sigma": 0.1,
                            "prob": 0.3
                        }
                    },
                    "restart_strategy": {
                        "Periodic": {
                            "frequency": 25
                        }
                    },
                    "intensification_cycles": 5,
                    "diversification_threshold": 0.1,
                    "success_history_size": 20,
                    "adaptation_rate": 0.1
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

    let mut tabu: TabuSearch<f64, U1, U2> = TabuSearch::new(tabu_conf, init_x, opt_prob);
    let initial_fitness = tabu.st.best_f;

    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.st.best_f > initial_fitness);
    assert!(tabu.st.best_x.iter().all(|&x| (-5.0..=5.0).contains(&x)));
}

#[test]
fn test_quality_based_tabu() {
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
                    "QualityBased": {
                        "quality_threshold": 0.1,
                        "quality_memory_size": 10
                    }
                },
                "advanced": {
                    "adaptive_parameters": true,
                    "aspiration_criteria": true,
                    "neighborhood_strategy": {
                        "Cauchy": {
                            "scale": 0.1,
                            "prob": 0.3
                        }
                    },
                    "restart_strategy": {
                        "Periodic": {
                            "frequency": 20
                        }
                    },
                    "intensification_cycles": 5,
                    "diversification_threshold": 0.1,
                    "success_history_size": 20,
                    "adaptation_rate": 0.1
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

    let mut tabu: TabuSearch<f64, U1, U2> = TabuSearch::new(tabu_conf, init_x, opt_prob);
    let initial_fitness = tabu.st.best_f;

    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.st.best_f > initial_fitness);
    assert!(tabu.st.best_x.iter().all(|&x| (-5.0..=5.0).contains(&x)));
}

#[test]
fn test_gaussian_neighborhood() {
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
                },
                "advanced": {
                    "adaptive_parameters": false,
                    "aspiration_criteria": false,
                    "neighborhood_strategy": {
                        "Gaussian": {
                            "sigma": 0.05,
                            "prob": 0.4
                        }
                    },
                    "restart_strategy": {
                        "None": null
                    },
                    "intensification_cycles": 5,
                    "diversification_threshold": 0.1,
                    "success_history_size": 20,
                    "adaptation_rate": 0.1
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

    let mut tabu: TabuSearch<f64, U1, U2> = TabuSearch::new(tabu_conf, init_x, opt_prob);
    let initial_fitness = tabu.st.best_f;

    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.st.best_f > initial_fitness);
    assert!(tabu.st.best_x.iter().all(|&x| (-5.0..=5.0).contains(&x)));
}

#[test]
fn test_adaptive_neighborhood() {
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
                },
                "advanced": {
                    "adaptive_parameters": true,
                    "aspiration_criteria": true,
                    "neighborhood_strategy": {
                        "Adaptive": {
                            "base_step": 0.1,
                            "adaptation_rate": 0.2
                        }
                    },
                    "restart_strategy": {
                        "Stagnation": {
                            "max_iterations": 30,
                            "threshold": 1e-6
                        }
                    },
                    "intensification_cycles": 3,
                    "diversification_threshold": 0.1,
                    "success_history_size": 20,
                    "adaptation_rate": 0.1
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

    let mut tabu: TabuSearch<f64, U1, U2> = TabuSearch::new(tabu_conf, init_x, opt_prob);
    let initial_fitness = tabu.st.best_f;

    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.st.best_f > initial_fitness);
    assert!(tabu.st.best_x.iter().all(|&x| (-5.0..=5.0).contains(&x)));
}

mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{OMatrix, OVector, U1, U10, U2, U5};

use non_convex_opt::utils::{
    config::{AlgConf, Config},
    opt_prob::{OptProb, OptimizationAlgorithm},
};

use non_convex_opt::algorithms::continous_ga::{
    cga::CGA,
    crossover::{CrossoverOperator, Heuristic, Random, SimulatedBinary},
    selection::{Residual, RouletteWheel, SelectionOperator, Tournament},
};

#[test]
fn test_roulette_wheel_selection() {
    let selection = RouletteWheel::new(10, 5);
    let population = OMatrix::<f64, U10, U5>::from_element_generic(U10, U5, 1.0);
    let fitness = OVector::<f64, U10>::from_element_generic(U10, U1, 1.0);
    let constraint = OVector::<bool, U10>::from_element_generic(U10, U1, true);
    let selected = selection.select(&population, &fitness, &constraint);
    assert_eq!(selected.nrows(), 5);
    assert_eq!(selected.ncols(), 5);
}

#[test]
fn test_tournament_selection() {
    let selection = Tournament::new(10, 5, 2);
    let population = OMatrix::<f64, U10, U5>::from_element_generic(U10, U5, 1.0);
    let fitness = OVector::<f64, U10>::from_element_generic(U10, U1, 1.0);
    let constraint = OVector::<bool, U10>::from_element_generic(U10, U1, true);
    let selected = selection.select(&population, &fitness, &constraint);
    assert_eq!(selected.nrows(), 5);
    assert_eq!(selected.ncols(), 5);
}

#[test]
fn test_residual_selection() {
    let selection = Residual::new(10, 5);
    let population = OMatrix::<f64, U10, U5>::from_element_generic(U10, U5, 1.0);
    let fitness = OVector::<f64, U10>::from_element_generic(U10, U1, 1.0);
    let constraint = OVector::<bool, U10>::from_element_generic(U10, U1, true);
    let selected = selection.select(&population, &fitness, &constraint);
    assert_eq!(selected.nrows(), 5);
    assert_eq!(selected.ncols(), 5);
}

#[test]
fn test_random_crossover() {
    let selection = RouletteWheel::new(10, 5);
    let population = OMatrix::<f64, U10, U5>::from_element_generic(U10, U5, 1.0);
    let fitness = OVector::<f64, U10>::from_element_generic(U10, U1, 1.0);
    let constraint = OVector::<bool, U10>::from_element_generic(U10, U1, true);
    let selected = selection.select(&population, &fitness, &constraint);
    let crossover = Random::new(0.9, 10);
    let offspring: OMatrix<f64, U10, U5> = crossover.crossover(&selected);
    assert_eq!(offspring.nrows(), 10);
    assert_eq!(offspring.ncols(), 5);
}

#[test]
fn test_heuristic_crossover() {
    let selection = RouletteWheel::new(10, 5);
    let population = OMatrix::<f64, U10, U5>::from_element_generic(U10, U5, 1.0);
    let fitness = OVector::<f64, U10>::from_element_generic(U10, U1, 1.0);
    let constraint = OVector::<bool, U10>::from_element_generic(U10, U1, true);
    let selected = selection.select(&population, &fitness, &constraint);
    let crossover = Heuristic::new(0.9, 10);
    let offspring: OMatrix<f64, U10, U5> = crossover.crossover(&selected);
    assert_eq!(offspring.nrows(), 10);
    assert_eq!(offspring.ncols(), 5);
}

#[test]
fn test_simulated_binary_crossover() {
    let selection = RouletteWheel::new(10, 5);
    let population = OMatrix::<f64, U10, U5>::from_element_generic(U10, U5, 1.0);
    let fitness = OVector::<f64, U10>::from_element_generic(U10, U1, 1.0);
    let constraint = OVector::<bool, U10>::from_element_generic(U10, U1, true);
    let selected = selection.select(&population, &fitness, &constraint);
    let crossover = SimulatedBinary::new(0.9, 15.0, 10);
    let offspring: OMatrix<f64, U10, U5> = crossover.crossover(&selected);
    assert_eq!(offspring.nrows(), 10);
    assert_eq!(offspring.ncols(), 5);
}

#[test]
fn test_adaptive_parameters() {
    let conf = Config::new(include_str!("jsons/cga.json")).unwrap();

    let cga_conf = match conf.alg_conf {
        AlgConf::CGA(cga_conf) => cga_conf,
        _ => panic!("Expected CGAConf"),
    };

    let pop_size = 10;

    let mut init_pop = OMatrix::zeros_generic(U10, U2);
    for i in 0..pop_size {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0; // Random values in [-2, 2]
        }
    }

    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let mut cga = CGA::new(cga_conf, init_pop, opt_prob, 10);

    let (initial_mut_rate, initial_cross_prob) = cga.get_current_parameters();
    
    // Run some steps to trigger adaptation
    for _ in 0..10 {
        cga.step();
    }

    let (final_mut_rate, final_cross_prob) = cga.get_current_parameters();
    
    // Parameters should be finite and within reasonable bounds
    assert!(final_mut_rate.is_finite());
    assert!(final_cross_prob.is_finite());
    assert!(final_mut_rate >= 0.001 && final_mut_rate <= 0.5);
    assert!(final_cross_prob >= 0.1 && final_cross_prob <= 0.95);
}

#[test]
fn test_cga() {
    let conf = Config::new(include_str!("jsons/cga.json")).unwrap();

    let cga_conf = match conf.alg_conf {
        AlgConf::CGA(cga_conf) => cga_conf,
        _ => panic!("Expected CGAConf"),
    };

    let pop_size = 10;

    let mut init_pop = OMatrix::zeros_generic(U10, U2);
    for i in 0..pop_size {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0; // Random values in [-2, 2]
        }
    }

    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let mut cga = CGA::new(cga_conf, init_pop, opt_prob, 5);

    for _ in 0..5 {
        cga.step();
    }

    assert!(cga.st.best_f.is_finite());
}

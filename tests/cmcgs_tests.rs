mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{OMatrix, OVector, U1, U10, U2};

use non_convex_opt::algorithms::cmcgs::cmcgs_opt::CMCGS;
use non_convex_opt::utils::{
    config::CMCGSConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_cmcgs() {
    let conf = CMCGSConf {
        max_depth: 3,
        simulation_count: 5,
        simulation_steps: 3,
        exploration_constant: 1.0,
        max_clusters: 5,
        max_policies: 3,
        merge_threshold: 0.5,
        initial_std: 0.1,
        restart_threshold: 50,
        expansion_threshold: 10,
        max_nodes_per_layer: 10,
        epsilon: 0.2,
        discount_factor: 0.9,
    };

    let init_x = OMatrix::<f64, U10, U2>::from_element_generic(U10, U2, 0.5);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut cmcgs: CMCGS<f64, U10, U2> = CMCGS::new(conf, init_x, opt_prob, 42);

    assert_eq!(cmcgs.iteration_count, 0);
    assert_eq!(cmcgs.graph.get_max_depth(), 0);

    for i in 0..5 {
        cmcgs.step();
        assert_eq!(cmcgs.iteration_count, i + 1);
        assert_eq!(cmcgs.st.iter, i + 1);

        assert!(
            cmcgs.st.best_x.iter().all(|&x| (0.0..=1.0).contains(&x)),
            "Best solution violated constraints: {:?}",
            cmcgs.st.best_x
        );

        assert_eq!(cmcgs.st.pop.nrows(), 10);
        assert_eq!(cmcgs.st.pop.ncols(), 2);
    }

    assert!(cmcgs.graph.get_max_depth() > 0);
}

#[test]
fn test_cmcgs_graph_operations() {
    let conf = CMCGSConf {
        max_depth: 4,
        simulation_count: 3,
        simulation_steps: 2,
        exploration_constant: 1.0,
        max_clusters: 4,
        max_policies: 2,
        merge_threshold: 0.5,
        initial_std: 0.1,
        restart_threshold: 100,
        expansion_threshold: 5,
        max_nodes_per_layer: 8,
        epsilon: 0.1,
        discount_factor: 0.9,
    };

    let init_x = OMatrix::<f64, U10, U2>::from_element_generic(U10, U2, 0.5);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut cmcgs: CMCGS<f64, U10, U2> = CMCGS::new(conf, init_x, opt_prob, 42);

    cmcgs.initialize_graph();

    assert!(cmcgs.graph.get_max_depth() > 0);
    assert!(!cmcgs.graph.get_root_id().is_empty());

    let root_id = cmcgs.graph.get_root_id()[0];
    assert!(cmcgs.graph.get_node(root_id).is_some());
}

#[test]
fn test_cmcgs_replay_buffer() {
    let conf = CMCGSConf {
        max_depth: 3,
        simulation_count: 2,
        simulation_steps: 2,
        exploration_constant: 1.0,
        max_clusters: 3,
        max_policies: 2,
        merge_threshold: 0.5,
        initial_std: 0.1,
        restart_threshold: 100,
        expansion_threshold: 5,
        max_nodes_per_layer: 5,
        epsilon: 0.1,
        discount_factor: 0.9,
    };

    let init_x = OMatrix::<f64, U10, U2>::from_element_generic(U10, U2, 0.5);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut cmcgs: CMCGS<f64, U10, U2> = CMCGS::new(conf, init_x, opt_prob, 42);

    for _ in 0..3 {
        cmcgs.step();
    }

    assert!(cmcgs.replay_buffer.len() > 0);

    let root_id = cmcgs.graph.get_root_id()[0];
    assert!(cmcgs.node_replay_buffers.get_buffer(root_id).is_some());
}

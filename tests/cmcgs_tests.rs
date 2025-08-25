mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{OMatrix, U1, U2};

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
        simulation_steps: 2,
        exploration_constant: 1.0,
        max_clusters: 5,
        max_policies: 3,
        merge_threshold: 0.5,
        initial_std: 0.1,
        restart_threshold: 50,
        expansion_threshold: 5,
        max_nodes_per_layer: 10,
        epsilon: 0.2,
        discount_factor: 0.9,
        top_experiences_count: 5,
        restart_max_attempts: 10,
    };

    let init_x = OMatrix::<f64, U1, U2>::from_element_generic(U1, U2, 1.0);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut cmcgs: CMCGS<f64, U1, U2> = CMCGS::new(conf, init_x.row(0).into_owned(), opt_prob, 42);

    for _ in 0..3 {
        cmcgs.step();

        assert!(
            cmcgs.st.best_x.iter().all(|&x| (-5.0..=5.0).contains(&x)),
            "Best solution violated bounds: {:?}",
            cmcgs.st.best_x
        );

        assert_eq!(cmcgs.st.pop.nrows(), 1);
        assert_eq!(cmcgs.st.pop.ncols(), 2);
    }

    assert!(cmcgs.graph.get_max_depth() > 1);
}

#[test]
fn test_cmcgs_graph() {
    let conf = CMCGSConf {
        max_depth: 3,
        simulation_count: 5,
        simulation_steps: 2,
        exploration_constant: 1.0,
        max_clusters: 4,
        max_policies: 2,
        merge_threshold: 0.5,
        initial_std: 0.1,
        restart_threshold: 100,
        expansion_threshold: 3,
        max_nodes_per_layer: 8,
        epsilon: 0.1,
        discount_factor: 0.9,
        top_experiences_count: 5,
        restart_max_attempts: 10,
    };

    let init_x = OMatrix::<f64, U1, U2>::from_element_generic(U1, U2, 1.0);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut cmcgs: CMCGS<f64, U1, U2> = CMCGS::new(conf, init_x.row(0).into_owned(), opt_prob, 42);

    cmcgs.step();

    assert!(cmcgs.graph.get_max_depth() > 1);
    let root_id = cmcgs.graph.get_root_id();
    assert!(cmcgs.graph.get_node(root_id).is_some());
}

#[test]
fn test_cmcgs_replay_buffer() {
    let conf = CMCGSConf {
        max_depth: 3,
        simulation_count: 5,
        simulation_steps: 2,
        exploration_constant: 1.0,
        max_clusters: 3,
        max_policies: 2,
        merge_threshold: 0.5,
        initial_std: 0.1,
        restart_threshold: 100,
        expansion_threshold: 3,
        max_nodes_per_layer: 5,
        epsilon: 0.1,
        discount_factor: 0.9,
        top_experiences_count: 5,
        restart_max_attempts: 10,
    };

    let init_x = OMatrix::<f64, U1, U2>::from_element_generic(U1, U2, 1.0);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut cmcgs: CMCGS<f64, U1, U2> = CMCGS::new(conf, init_x.row(0).into_owned(), opt_prob, 42);

    for _ in 0..2 {
        cmcgs.step();
    }

    assert!(!cmcgs.replay_buffer.is_empty());

    let root_id = cmcgs.graph.get_root_id();
    assert!(cmcgs.node_replay_buffers.get_buffer(root_id).is_some());
}

#[test]
fn test_cmcgs_state_clustering() {
    let conf = CMCGSConf {
        max_depth: 3,
        simulation_count: 5,
        simulation_steps: 2,
        exploration_constant: 1.0,
        max_clusters: 5,
        max_policies: 2,
        merge_threshold: 0.5,
        initial_std: 0.1,
        restart_threshold: 100,
        expansion_threshold: 3,
        max_nodes_per_layer: 8,
        epsilon: 0.1,
        discount_factor: 0.9,
        top_experiences_count: 5,
        restart_max_attempts: 10,
    };

    let init_x = OMatrix::<f64, U1, U2>::from_element_generic(U1, U2, 1.0);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut cmcgs: CMCGS<f64, U1, U2> =
        CMCGS::new(conf.clone(), init_x.row(0).into_owned(), opt_prob, 42);

    cmcgs.step();
    let initial_layer_count = cmcgs.graph.get_max_depth();

    for _ in 0..4 {
        cmcgs.step();
    }

    let final_layer_count = cmcgs.graph.get_max_depth();
    assert!(
        final_layer_count >= initial_layer_count,
        "Expected graph depth to increase or stay same, got {} -> {}",
        initial_layer_count,
        final_layer_count
    );

    for layer in 0..final_layer_count {
        let layer_size = cmcgs.graph.get_layer_size(layer);
        assert!(layer_size > 0, "Layer {} should have nodes", layer);
        assert!(
            layer_size <= conf.max_nodes_per_layer,
            "Layer {} should not exceed max_nodes_per_layer",
            layer
        );
    }

    assert!(
        cmcgs.graph.size() > 1,
        "Graph should have multiple nodes after clustering"
    );
}

#[test]
fn test_cmcgs_gaussian_policy_updates() {
    let conf = CMCGSConf {
        max_depth: 3,
        simulation_count: 5,
        simulation_steps: 2,
        exploration_constant: 1.0,
        max_clusters: 4,
        max_policies: 3,
        merge_threshold: 0.5,
        initial_std: 0.1,
        restart_threshold: 100,
        expansion_threshold: 4,
        max_nodes_per_layer: 6,
        epsilon: 0.2,
        discount_factor: 0.9,
        top_experiences_count: 5,
        restart_max_attempts: 10,
    };

    let init_x = OMatrix::<f64, U1, U2>::from_element_generic(U1, U2, 1.0);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut cmcgs: CMCGS<f64, U1, U2> = CMCGS::new(conf, init_x.row(0).into_owned(), opt_prob, 42);

    cmcgs.step();
    let root_id = cmcgs.graph.get_root_id();

    let initial_node = cmcgs.graph.get_node(root_id).unwrap();
    let initial_action_policy = initial_node.get_action_policy();
    let initial_mean = initial_action_policy.mean;
    let initial_variance = initial_action_policy.variance;

    for _ in 0..5 {
        cmcgs.step();
    }

    let current_root_id = cmcgs.graph.get_root_id();
    let updated_node = cmcgs.graph.get_node(current_root_id).unwrap();
    let updated_action_policy = updated_node.get_action_policy();
    let updated_mean = updated_action_policy.mean;
    let updated_variance = updated_action_policy.variance;

    let mean_changed = initial_mean
        .iter()
        .zip(updated_mean.iter())
        .any(|(init, updated)| (init - updated).abs() > 1e-6);
    assert!(
        mean_changed,
        "Action policy mean should change during learning"
    );

    let variance_changed = initial_variance
        .iter()
        .zip(updated_variance.iter())
        .any(|(init, updated)| (init - updated).abs() > 1e-6);
    assert!(
        variance_changed,
        "Action policy variance should change during learning"
    );

    let root_buffer = cmcgs.node_replay_buffers.get_buffer(current_root_id);
    assert!(root_buffer.is_some(), "Root node should have replay buffer");
    if let Some(buffer) = root_buffer {
        assert!(
            !buffer.is_empty(),
            "Root node replay buffer should contain experiences"
        );
    }

    let final_node = cmcgs.graph.get_node(current_root_id).unwrap();
    assert!(final_node.visits > 0, "Root node should have been visited");
    assert!(
        final_node.total_reward > f64::NEG_INFINITY,
        "Root node should have accumulated rewards"
    );
}

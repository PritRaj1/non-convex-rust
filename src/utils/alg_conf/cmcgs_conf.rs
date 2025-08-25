use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CMCGSConf {
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    #[serde(default = "default_simulation_count")]
    pub simulation_count: usize,
    #[serde(default = "default_simulation_steps")]
    pub simulation_steps: usize,
    #[serde(default = "default_exploration_constant")]
    pub exploration_constant: f64,
    #[serde(default = "default_max_clusters")]
    pub max_clusters: usize,
    #[serde(default = "default_max_policies")]
    pub max_policies: usize,
    #[serde(default = "default_merge_threshold")]
    pub merge_threshold: f64,
    #[serde(default = "default_initial_std")]
    pub initial_std: f64,

    #[serde(default = "default_restart_threshold")]
    pub restart_threshold: usize,
    #[serde(default = "default_expansion_threshold")]
    pub expansion_threshold: usize,
    #[serde(default = "default_max_nodes_per_layer")]
    pub max_nodes_per_layer: usize,
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    #[serde(default = "default_discount_factor")]
    pub discount_factor: f64,
    #[serde(default = "default_top_experiences_count")]
    pub top_experiences_count: usize,
    #[serde(default = "default_restart_max_attempts")]
    pub restart_max_attempts: usize,
}

fn default_max_depth() -> usize {
    5
}
fn default_simulation_count() -> usize {
    10
}
fn default_simulation_steps() -> usize {
    5
}
fn default_exploration_constant() -> f64 {
    1.414
}
fn default_max_clusters() -> usize {
    20
}
fn default_max_policies() -> usize {
    15
}
fn default_merge_threshold() -> f64 {
    0.5
}
fn default_initial_std() -> f64 {
    1.0
}

fn default_restart_threshold() -> usize {
    20
}
fn default_expansion_threshold() -> usize {
    50
}
fn default_max_nodes_per_layer() -> usize {
    20
}
fn default_epsilon() -> f64 {
    0.1
}
fn default_discount_factor() -> f64 {
    0.99
}
fn default_top_experiences_count() -> usize {
    5
}
fn default_restart_max_attempts() -> usize {
    100
}

impl Default for CMCGSConf {
    fn default() -> Self {
        Self {
            max_depth: default_max_depth(),
            simulation_count: default_simulation_count(),
            simulation_steps: default_simulation_steps(),
            exploration_constant: default_exploration_constant(),
            max_clusters: default_max_clusters(),
            max_policies: default_max_policies(),
            merge_threshold: default_merge_threshold(),
            initial_std: default_initial_std(),

            restart_threshold: default_restart_threshold(),
            expansion_threshold: default_expansion_threshold(),
            max_nodes_per_layer: default_max_nodes_per_layer(),
            epsilon: default_epsilon(),
            discount_factor: default_discount_factor(),
            top_experiences_count: default_top_experiences_count(),
            restart_max_attempts: default_restart_max_attempts(),
        }
    }
}

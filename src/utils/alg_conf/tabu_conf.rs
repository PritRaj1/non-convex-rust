use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TabuConf {
    pub common: CommonConf,
    pub list_type: ListType,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum ListType {
    Standard(StandardConf),
    Reactive(ReactiveConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CommonConf {
    #[serde(default = "default_tabu_list_size")]
    pub tabu_list_size: usize,
    #[serde(default = "default_num_neighbors")]
    pub num_neighbors: usize,
    #[serde(default = "default_step_size")]
    pub step_size: f64,
    #[serde(default = "default_perturbation_prob")]
    pub perturbation_prob: f64,
    #[serde(default = "default_tabu_threshold")]
    pub tabu_threshold: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ReactiveConf {
    #[serde(default = "default_min_tabu_size")]
    pub min_tabu_size: usize,
    #[serde(default = "default_max_tabu_size")]
    pub max_tabu_size: usize,
    #[serde(default = "default_increase_factor")]
    pub increase_factor: f64,
    #[serde(default = "default_decrease_factor")]
    pub decrease_factor: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct StandardConf {}

fn default_tabu_list_size() -> usize {
    20
}
fn default_num_neighbors() -> usize {
    50
}
fn default_step_size() -> f64 {
    0.1
}
fn default_perturbation_prob() -> f64 {
    0.3
}
fn default_tabu_threshold() -> f64 {
    1e-6
}
fn default_min_tabu_size() -> usize {
    10
}
fn default_max_tabu_size() -> usize {
    30
}
fn default_increase_factor() -> f64 {
    1.1
}
fn default_decrease_factor() -> f64 {
    0.9
}

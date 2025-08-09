use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SAConf {
    #[serde(default = "default_initial_temp")]
    pub initial_temp: f64,
    #[serde(default = "default_cooling_rate")]
    pub cooling_rate: f64,
    #[serde(default = "default_step_size")]
    pub step_size: f64,
    #[serde(default = "default_num_neighbors")]
    pub num_neighbors: usize,
    #[serde(default = "default_reheat_after")]
    pub reheat_after: usize,
    #[serde(default = "default_x_min")]
    pub x_min: f64,
    #[serde(default = "default_x_max")]
    pub x_max: f64,
}

fn default_initial_temp() -> f64 {
    1000.0
}
fn default_cooling_rate() -> f64 {
    0.998
}
fn default_step_size() -> f64 {
    0.5
}
fn default_num_neighbors() -> usize {
    20
}
fn default_reheat_after() -> usize {
    50
}
fn default_x_min() -> f64 {
    -10.0
}
fn default_x_max() -> f64 {
    10.0
}

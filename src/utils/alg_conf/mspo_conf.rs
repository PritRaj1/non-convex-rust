use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MSPOConf {
    #[serde(default = "default_num_swarms")]
    pub num_swarms: usize,
    #[serde(default = "default_swarm_size")]
    pub swarm_size: usize,
    #[serde(default = "default_c1")]
    pub c1: f64, // Cognitive coefficient
    #[serde(default = "default_c2")]
    pub c2: f64, // Social coefficient
    #[serde(default = "default_x_min")]
    pub x_min: f64, // Lower bound for initialization
    #[serde(default = "default_x_max")]
    pub x_max: f64, // Upper bound for initialization
    #[serde(default = "default_exchange_interval")]
    pub exchange_interval: usize, // How often to exchange information between swarms
    #[serde(default = "default_exchange_ratio")]
    pub exchange_ratio: f64, // Fraction of particles to exchange information
    #[serde(default = "default_improvement_threshold")]
    pub improvement_threshold: f64, // Minimum relative improvement needed for exchange
    #[serde(default = "default_inertia_start")]
    pub inertia_start: f64, // Starting inertia weight for adaptive calculation
    #[serde(default = "default_inertia_end")]
    pub inertia_end: f64, // Ending inertia weight for adaptive calculation
}

fn default_num_swarms() -> usize {
    5
}
fn default_swarm_size() -> usize {
    50
}

fn default_c1() -> f64 {
    2.05
}
fn default_c2() -> f64 {
    2.05
}
fn default_x_min() -> f64 {
    -10.0
}
fn default_x_max() -> f64 {
    10.0
}
fn default_exchange_interval() -> usize {
    10
}
fn default_exchange_ratio() -> f64 {
    0.1
}
fn default_improvement_threshold() -> f64 {
    0.1
} // 10% improvement needed by default

fn default_inertia_start() -> f64 {
    0.9
}

fn default_inertia_end() -> f64 {
    0.4
}

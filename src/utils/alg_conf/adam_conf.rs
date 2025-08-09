use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdamConf {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_beta1")]
    pub beta1: f64,
    #[serde(default = "default_beta2")]
    pub beta2: f64,
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
}

fn default_learning_rate() -> f64 {
    0.001
}
fn default_beta1() -> f64 {
    0.9
}
fn default_beta2() -> f64 {
    0.999
}
fn default_epsilon() -> f64 {
    1e-8
}

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
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_gradient_clip")]
    pub gradient_clip: f64,
    #[serde(default = "default_amsgrad")]
    pub amsgrad: bool,
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
fn default_weight_decay() -> f64 {
    0.0
}
fn default_gradient_clip() -> f64 {
    1.0
}
fn default_amsgrad() -> bool {
    false
}

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SGAConf {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_momentum")]
    pub momentum: f64,
    #[serde(default = "default_gradient_clip")]
    pub gradient_clip: f64,
    #[serde(default = "default_noise_decay")]
    pub noise_decay: f64,
    #[serde(default = "default_adaptive_noise")]
    pub adaptive_noise: bool,
}

fn default_learning_rate() -> f64 {
    0.01
}
fn default_momentum() -> f64 {
    0.9
}
fn default_gradient_clip() -> f64 {
    1.0
}
fn default_noise_decay() -> f64 {
    0.99
}
fn default_adaptive_noise() -> bool {
    false
}

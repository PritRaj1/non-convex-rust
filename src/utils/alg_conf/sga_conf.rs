use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SGAConf {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_momentum")]
    pub momentum: f64,
}

fn default_learning_rate() -> f64 {
    0.01
}
fn default_momentum() -> f64 {
    0.9
}

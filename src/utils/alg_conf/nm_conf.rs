use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NelderMeadConf {
    #[serde(default = "default_alpha")]
    pub alpha: f64, // Reflection coefficient
    #[serde(default = "default_gamma")]
    pub gamma: f64, // Expansion coefficient
    #[serde(default = "default_rho")]
    pub rho: f64, // Contraction coefficient
    #[serde(default = "default_sigma")]
    pub sigma: f64, // Shrink coefficient
}

fn default_alpha() -> f64 {
    1.0
}
fn default_gamma() -> f64 {
    2.0
}
fn default_rho() -> f64 {
    0.5
}
fn default_sigma() -> f64 {
    0.5
}

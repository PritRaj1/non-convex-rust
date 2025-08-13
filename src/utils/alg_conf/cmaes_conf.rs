use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CMAESConf {
    #[serde(default = "default_num_parents")]
    pub num_parents: usize,
    #[serde(default = "default_initial_sigma")]
    pub initial_sigma: f64,
    #[serde(default = "default_use_active_cma")]
    pub use_active_cma: bool,
    #[serde(default = "default_active_cma_ratio")]
    pub active_cma_ratio: f64,
}

fn default_num_parents() -> usize {
    50
}
fn default_initial_sigma() -> f64 {
    0.3
}
fn default_use_active_cma() -> bool {
    true
}
fn default_active_cma_ratio() -> f64 {
    0.25
}

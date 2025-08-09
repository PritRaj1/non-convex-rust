use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CMAESConf {
    #[serde(default = "default_num_parents")]
    pub num_parents: usize,
    #[serde(default = "default_initial_sigma")]
    pub initial_sigma: f64,
}

fn default_num_parents() -> usize {
    50
}
fn default_initial_sigma() -> f64 {
    0.3
}

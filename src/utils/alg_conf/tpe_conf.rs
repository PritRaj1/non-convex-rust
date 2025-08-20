use serde::{Deserialize, Serialize};

pub use crate::algorithms::tpe::kernels::KernelType;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TPEConf {
    #[serde(default = "default_n_initial_random")]
    pub n_initial_random: usize,
    #[serde(default = "default_n_ei_candidates")]
    pub n_ei_candidates: usize,
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    #[serde(default = "default_prior_weight")]
    pub prior_weight: f64,
    #[serde(default = "default_kernel_type")]
    pub kernel_type: KernelType,
    #[serde(default = "default_max_history")]
    pub max_history: usize,
    #[serde(default = "default_advanced")]
    pub advanced: AdvancedConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdvancedConf {
    #[serde(default = "default_use_restart_strategy")]
    pub use_restart_strategy: bool,
    #[serde(default = "default_restart_frequency")]
    pub restart_frequency: usize,
}

fn default_n_initial_random() -> usize {
    20
}

fn default_n_ei_candidates() -> usize {
    100
}

fn default_gamma() -> f64 {
    0.25
}

fn default_prior_weight() -> f64 {
    1.0
}

fn default_kernel_type() -> KernelType {
    KernelType::Gaussian
}

fn default_max_history() -> usize {
    1000
}

fn default_use_restart_strategy() -> bool {
    false
}

fn default_restart_frequency() -> usize {
    100
}

fn default_advanced() -> AdvancedConf {
    AdvancedConf {
        use_restart_strategy: default_use_restart_strategy(),
        restart_frequency: default_restart_frequency(),
    }
}

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PTConf {
    pub common: CommonConf,
    pub swap_conf: SwapConf,
    pub update_conf: UpdateConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CommonConf {
    #[serde(default = "default_num_replicas")]
    pub num_replicas: usize,
    #[serde(default = "default_power_law_init")]
    pub power_law_init: f64,
    #[serde(default = "default_power_law_final")]
    pub power_law_final: f64,
    #[serde(default = "default_power_law_cycles")]
    pub power_law_cycles: usize,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    #[serde(default = "default_omega")]
    pub omega: f64,
    #[serde(default = "default_mala_step_size")]
    pub mala_step_size: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum SwapConf {
    Periodic(PeriodicConf),
    Stochastic(StochasticConf),
    Always(AlwaysConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum UpdateConf {
    MetropolisHastings(MetropolisHastingsConf),
    MALA(MALAConf),
    PCN(PCNConf),
    Auto(AutoConf), // Automatically choose based on gradient availability
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MetropolisHastingsConf {
    #[serde(default = "default_random_walk_step_size")]
    pub random_walk_step_size: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MALAConf {
    #[serde(default = "default_mala_step_size")]
    pub step_size: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PCNConf {
    #[serde(default = "default_pcn_step_size")]
    pub step_size: f64,
    #[serde(default = "default_pcn_preconditioner")]
    pub preconditioner: f64, // Preconditioning factor
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AlwaysConf {}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PeriodicConf {
    #[serde(default = "default_swap_frequency")]
    pub swap_frequency: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct StochasticConf {
    #[serde(default = "default_swap_probability")]
    pub swap_probability: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AutoConf {} // No additional parameters needed

impl PTConf {
    pub fn new() -> Self {
        Self {
            common: CommonConf::new(),
            swap_conf: SwapConf::Always(AlwaysConf {}),
            update_conf: UpdateConf::Auto(AutoConf {}),
        }
    }
}

impl Default for PTConf {
    fn default() -> Self {
        Self::new()
    }
}

impl CommonConf {
    pub fn new() -> Self {
        Self {
            num_replicas: default_num_replicas(),
            power_law_init: default_power_law_init(),
            power_law_final: default_power_law_final(),
            power_law_cycles: default_power_law_cycles(),
            alpha: default_alpha(),
            omega: default_omega(),
            mala_step_size: default_mala_step_size(),
        }
    }
}

impl Default for CommonConf {
    fn default() -> Self {
        Self::new()
    }
}

fn default_num_replicas() -> usize {
    10
}
fn default_power_law_init() -> f64 {
    2.0
}
fn default_power_law_final() -> f64 {
    0.5
}
fn default_power_law_cycles() -> usize {
    1
}
fn default_alpha() -> f64 {
    0.1
}
fn default_omega() -> f64 {
    2.1
}
fn default_mala_step_size() -> f64 {
    0.01
}
fn default_swap_frequency() -> f64 {
    1.0
}
fn default_swap_probability() -> f64 {
    0.1
}
fn default_random_walk_step_size() -> f64 {
    0.1
}
fn default_pcn_step_size() -> f64 {
    0.01
}
fn default_pcn_preconditioner() -> f64 {
    1.0
}

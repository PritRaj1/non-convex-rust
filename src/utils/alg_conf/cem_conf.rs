use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CEMConf {
    pub common: CommonConf,
    pub sampling: SamplingConf,
    pub adaptation: AdaptationConf,
    pub advanced: AdvancedConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CommonConf {
    #[serde(default = "default_population_size")]
    pub population_size: usize,
    #[serde(default = "default_elite_size")]
    pub elite_size: usize,
    #[serde(default = "default_initial_std")]
    pub initial_std: f64,
    #[serde(default = "default_min_std")]
    pub min_std: f64,
    #[serde(default = "default_max_std")]
    pub max_std: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SamplingConf {
    #[serde(default = "default_use_antithetic")]
    pub use_antithetic: bool,
    #[serde(default = "default_antithetic_ratio")]
    pub antithetic_ratio: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdaptationConf {
    #[serde(default = "default_smoothing_factor")]
    pub smoothing_factor: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdvancedConf {
    #[serde(default = "default_use_restart_strategy")]
    pub use_restart_strategy: bool,
    #[serde(default = "default_restart_frequency")]
    pub restart_frequency: usize,

    #[serde(default = "default_use_covariance_adaptation")]
    pub use_covariance_adaptation: bool,
    #[serde(default = "default_covariance_regularization")]
    pub covariance_regularization: f64,
    #[serde(default = "default_improvement_history_size")]
    pub improvement_history_size: usize,
    #[serde(default = "default_improvement_threshold_window")]
    pub improvement_threshold_window: usize,
}

impl CEMConf {
    pub fn new() -> Self {
        Self {
            common: CommonConf::new(),
            sampling: SamplingConf::new(),
            adaptation: AdaptationConf::new(),
            advanced: AdvancedConf::new(),
        }
    }
}

impl Default for CEMConf {
    fn default() -> Self {
        Self::new()
    }
}

impl CommonConf {
    pub fn new() -> Self {
        Self {
            population_size: default_population_size(),
            elite_size: default_elite_size(),
            initial_std: default_initial_std(),
            min_std: default_min_std(),
            max_std: default_max_std(),
        }
    }
}

impl Default for CommonConf {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingConf {
    pub fn new() -> Self {
        Self {
            use_antithetic: default_use_antithetic(),
            antithetic_ratio: default_antithetic_ratio(),
        }
    }
}

impl Default for SamplingConf {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptationConf {
    pub fn new() -> Self {
        Self {
            smoothing_factor: default_smoothing_factor(),
        }
    }
}

impl Default for AdaptationConf {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedConf {
    pub fn new() -> Self {
        Self {
            use_restart_strategy: default_use_restart_strategy(),
            restart_frequency: default_restart_frequency(),
            use_covariance_adaptation: default_use_covariance_adaptation(),
            covariance_regularization: default_covariance_regularization(),
            improvement_history_size: default_improvement_history_size(),
            improvement_threshold_window: default_improvement_threshold_window(),
        }
    }
}

impl Default for AdvancedConf {
    fn default() -> Self {
        Self::new()
    }
}

fn default_population_size() -> usize {
    100
}

fn default_elite_size() -> usize {
    20
}

fn default_initial_std() -> f64 {
    1.0
}

fn default_min_std() -> f64 {
    1e-6
}

fn default_max_std() -> f64 {
    10.0
}

fn default_use_antithetic() -> bool {
    false
}

fn default_antithetic_ratio() -> f64 {
    0.5
}

fn default_smoothing_factor() -> f64 {
    0.7
}

fn default_use_restart_strategy() -> bool {
    true
}

fn default_restart_frequency() -> usize {
    100
}

fn default_use_covariance_adaptation() -> bool {
    true
}

fn default_covariance_regularization() -> f64 {
    1e-6
}

fn default_improvement_history_size() -> usize {
    100
}

fn default_improvement_threshold_window() -> usize {
    20
}

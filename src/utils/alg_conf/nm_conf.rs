use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct NelderMeadConf {
    pub common: CommonConf,
    pub advanced: AdvancedConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CommonConf {
    #[serde(default = "default_alpha")]
    pub alpha: f64, // Reflection coefficient
    #[serde(default = "default_gamma")]
    pub gamma: f64, // Expansion coefficient
    #[serde(default = "default_rho")]
    pub rho: f64, // Contraction coefficient
    #[serde(default = "default_sigma")]
    pub sigma: f64, // Shrink coefficient
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdvancedConf {
    #[serde(default = "default_adaptive_parameters")]
    pub adaptive_parameters: bool,
    #[serde(default = "default_restart_strategy")]
    pub restart_strategy: RestartStrategy,
    #[serde(default = "default_stagnation_detection")]
    pub stagnation_detection: StagnationDetection,
    #[serde(default = "default_coefficient_bounds")]
    pub coefficient_bounds: CoefficientBounds,
    #[serde(default = "default_adaptation_rate")]
    pub adaptation_rate: f64,
    #[serde(default = "default_success_history_size")]
    pub success_history_size: usize,
    #[serde(default = "default_improvement_history_size")]
    pub improvement_history_size: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum RestartStrategy {
    None,
    Periodic {
        frequency: usize,
    },
    Stagnation {
        max_iterations: usize,
        threshold: f64,
    },
    Adaptive {
        base_frequency: usize,
        adaptation_rate: f64,
    },
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct StagnationDetection {
    #[serde(default = "default_stagnation_window")]
    pub stagnation_window: usize,
    #[serde(default = "default_improvement_threshold")]
    pub improvement_threshold: f64,
    #[serde(default = "default_diversity_threshold")]
    pub diversity_threshold: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CoefficientBounds {
    #[serde(default = "default_alpha_bounds")]
    pub alpha_bounds: (f64, f64),
    #[serde(default = "default_gamma_bounds")]
    pub gamma_bounds: (f64, f64),
    #[serde(default = "default_rho_bounds")]
    pub rho_bounds: (f64, f64),
    #[serde(default = "default_sigma_bounds")]
    pub sigma_bounds: (f64, f64),
}

impl Default for CommonConf {
    fn default() -> Self {
        Self {
            alpha: default_alpha(),
            gamma: default_gamma(),
            rho: default_rho(),
            sigma: default_sigma(),
        }
    }
}

impl Default for AdvancedConf {
    fn default() -> Self {
        Self {
            adaptive_parameters: default_adaptive_parameters(),
            restart_strategy: default_restart_strategy(),
            stagnation_detection: default_stagnation_detection(),
            coefficient_bounds: default_coefficient_bounds(),
            adaptation_rate: default_adaptation_rate(),
            success_history_size: default_success_history_size(),
            improvement_history_size: default_improvement_history_size(),
        }
    }
}

impl Default for RestartStrategy {
    fn default() -> Self {
        Self::Stagnation {
            max_iterations: 50,
            threshold: 1e-6,
        }
    }
}

impl Default for StagnationDetection {
    fn default() -> Self {
        Self {
            stagnation_window: 20,
            improvement_threshold: 1e-6,
            diversity_threshold: 1e-3,
        }
    }
}

impl Default for CoefficientBounds {
    fn default() -> Self {
        Self {
            alpha_bounds: default_alpha_bounds(),
            gamma_bounds: default_gamma_bounds(),
            rho_bounds: default_rho_bounds(),
            sigma_bounds: default_sigma_bounds(),
        }
    }
}

// Default value functions
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

fn default_adaptive_parameters() -> bool {
    true
}
fn default_restart_strategy() -> RestartStrategy {
    RestartStrategy::default()
}
fn default_stagnation_detection() -> StagnationDetection {
    StagnationDetection::default()
}
fn default_coefficient_bounds() -> CoefficientBounds {
    CoefficientBounds::default()
}
fn default_adaptation_rate() -> f64 {
    0.1
}
fn default_success_history_size() -> usize {
    20
}
fn default_improvement_history_size() -> usize {
    30
}

fn default_alpha_bounds() -> (f64, f64) {
    (0.1, 3.0)
}
fn default_gamma_bounds() -> (f64, f64) {
    (1.0, 5.0)
}
fn default_rho_bounds() -> (f64, f64) {
    (0.1, 1.0)
}
fn default_sigma_bounds() -> (f64, f64) {
    (0.1, 1.0)
}

fn default_stagnation_window() -> usize {
    20
}
fn default_improvement_threshold() -> f64 {
    1e-6
}
fn default_diversity_threshold() -> f64 {
    1e-3
}

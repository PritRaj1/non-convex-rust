use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SAConf {
    #[serde(default = "default_initial_temp")]
    pub initial_temp: f64,
    #[serde(default = "default_cooling_rate")]
    pub cooling_rate: f64,
    #[serde(default = "default_step_size")]
    pub step_size: f64,
    #[serde(default = "default_num_neighbors")]
    pub num_neighbors: usize,
    #[serde(default = "default_x_min")]
    pub x_min: f64,
    #[serde(default = "default_x_max")]
    pub x_max: f64,
    #[serde(default = "default_min_step_size_factor")]
    pub min_step_size_factor: f64,
    #[serde(default = "default_step_size_decay_power")]
    pub step_size_decay_power: f64,
    #[serde(default = "default_min_temp_factor")]
    pub min_temp_factor: f64,
    #[serde(default = "default_use_adaptive_cooling")]
    pub use_adaptive_cooling: bool,
    #[serde(default = "default_advanced")]
    pub advanced: AdvancedConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdvancedConf {
    #[serde(default = "default_restart_strategy")]
    pub restart_strategy: RestartStrategy,
    #[serde(default = "default_stagnation_detection")]
    pub stagnation_detection: StagnationDetection,
    #[serde(default = "default_adaptive_parameters")]
    pub adaptive_parameters: bool,
    #[serde(default = "default_adaptation_rate")]
    pub adaptation_rate: f64,
    #[serde(default = "default_improvement_history_size")]
    pub improvement_history_size: usize,
    #[serde(default = "default_success_history_size")]
    pub success_history_size: usize,
    #[serde(default = "default_cooling_schedule")]
    pub cooling_schedule: CoolingScheduleType,
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
    Diversity {
        min_diversity: f64,
    },
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct StagnationDetection {
    #[serde(default = "default_stagnation_window")]
    pub stagnation_window: usize,
    #[serde(default = "default_improvement_threshold")]
    pub improvement_threshold: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum CoolingScheduleType {
    Exponential,
    Logarithmic,
    Cauchy,
    Adaptive,
}

fn default_initial_temp() -> f64 {
    1000.0
}
fn default_cooling_rate() -> f64 {
    0.998
}
fn default_step_size() -> f64 {
    0.5
}
fn default_num_neighbors() -> usize {
    20
}
fn default_x_min() -> f64 {
    -10.0
}
fn default_x_max() -> f64 {
    10.0
}
fn default_min_step_size_factor() -> f64 {
    0.1
}
fn default_step_size_decay_power() -> f64 {
    0.5
}
fn default_min_temp_factor() -> f64 {
    0.1
}

fn default_use_adaptive_cooling() -> bool {
    true
}

fn default_advanced() -> AdvancedConf {
    AdvancedConf {
        restart_strategy: default_restart_strategy(),
        stagnation_detection: default_stagnation_detection(),
        adaptive_parameters: default_adaptive_parameters(),
        adaptation_rate: default_adaptation_rate(),
        improvement_history_size: default_improvement_history_size(),
        success_history_size: default_success_history_size(),
        cooling_schedule: default_cooling_schedule(),
    }
}

fn default_restart_strategy() -> RestartStrategy {
    RestartStrategy::Stagnation {
        max_iterations: 100,
        threshold: 1e-6,
    }
}

fn default_stagnation_detection() -> StagnationDetection {
    StagnationDetection {
        stagnation_window: 10,
        improvement_threshold: 1e-6,
    }
}

fn default_adaptive_parameters() -> bool {
    true
}

fn default_adaptation_rate() -> f64 {
    0.1
}

fn default_improvement_history_size() -> usize {
    20
}

fn default_success_history_size() -> usize {
    20
}

fn default_cooling_schedule() -> CoolingScheduleType {
    CoolingScheduleType::Exponential
}

fn default_stagnation_window() -> usize {
    10
}

fn default_improvement_threshold() -> f64 {
    1e-6
}

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TabuConf {
    pub common: CommonConf,
    pub list_type: ListType,
    pub advanced: AdvancedConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum ListType {
    Standard(StandardConf),
    Reactive(ReactiveConf),
    FrequencyBased(FrequencyBasedConf),
    QualityBased(QualityBasedConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CommonConf {
    #[serde(default = "default_tabu_list_size")]
    pub tabu_list_size: usize,
    #[serde(default = "default_num_neighbors")]
    pub num_neighbors: usize,
    #[serde(default = "default_step_size")]
    pub step_size: f64,
    #[serde(default = "default_perturbation_prob")]
    pub perturbation_prob: f64,
    #[serde(default = "default_tabu_threshold")]
    pub tabu_threshold: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ReactiveConf {
    #[serde(default = "default_min_tabu_size")]
    pub min_tabu_size: usize,
    #[serde(default = "default_max_tabu_size")]
    pub max_tabu_size: usize,
    #[serde(default = "default_increase_factor")]
    pub increase_factor: f64,
    #[serde(default = "default_decrease_factor")]
    pub decrease_factor: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct FrequencyBasedConf {
    #[serde(default = "default_frequency_threshold")]
    pub frequency_threshold: usize,
    #[serde(default = "default_max_frequency")]
    pub max_frequency: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct QualityBasedConf {
    #[serde(default = "default_quality_threshold")]
    pub quality_threshold: f64,
    #[serde(default = "default_quality_memory_size")]
    pub quality_memory_size: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct StandardConf {}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdvancedConf {
    #[serde(default = "default_adaptive_parameters")]
    pub adaptive_parameters: bool,
    #[serde(default = "default_aspiration_criteria")]
    pub aspiration_criteria: bool,
    #[serde(default = "default_neighborhood_strategy")]
    pub neighborhood_strategy: NeighborhoodStrategy,
    #[serde(default = "default_restart_strategy")]
    pub restart_strategy: RestartStrategy,
    #[serde(default = "default_intensification_cycles")]
    pub intensification_cycles: usize,
    #[serde(default = "default_diversification_threshold")]
    pub diversification_threshold: f64,
    #[serde(default = "default_success_history_size")]
    pub success_history_size: usize,
    #[serde(default = "default_adaptation_rate")]
    pub adaptation_rate: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum NeighborhoodStrategy {
    Uniform {
        step_size: f64,
        prob: f64,
    },
    Gaussian {
        sigma: f64,
        prob: f64,
    },
    Cauchy {
        scale: f64,
        prob: f64,
    },
    Adaptive {
        base_step: f64,
        adaptation_rate: f64,
    },
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

fn default_tabu_list_size() -> usize {
    20
}
fn default_num_neighbors() -> usize {
    50
}
fn default_step_size() -> f64 {
    0.1
}
fn default_perturbation_prob() -> f64 {
    0.3
}
fn default_tabu_threshold() -> f64 {
    1e-6
}
fn default_min_tabu_size() -> usize {
    10
}
fn default_max_tabu_size() -> usize {
    30
}
fn default_increase_factor() -> f64 {
    1.1
}
fn default_decrease_factor() -> f64 {
    0.9
}
fn default_frequency_threshold() -> usize {
    3
}
fn default_max_frequency() -> usize {
    10
}
fn default_quality_threshold() -> f64 {
    0.1
}
fn default_quality_memory_size() -> usize {
    50
}
fn default_adaptive_parameters() -> bool {
    true
}
fn default_aspiration_criteria() -> bool {
    true
}
fn default_neighborhood_strategy() -> NeighborhoodStrategy {
    NeighborhoodStrategy::Uniform {
        step_size: 0.1,
        prob: 0.3,
    }
}
fn default_restart_strategy() -> RestartStrategy {
    RestartStrategy::Stagnation {
        max_iterations: 100,
        threshold: 1e-6,
    }
}
fn default_intensification_cycles() -> usize {
    5
}
fn default_diversification_threshold() -> f64 {
    0.1
}
fn default_success_history_size() -> usize {
    20
}
fn default_adaptation_rate() -> f64 {
    0.1
}

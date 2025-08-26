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
    #[serde(default = "default_kde_refit_frequency")]
    pub kde_refit_frequency: usize,
    #[serde(default = "default_advanced")]
    pub advanced: AdvancedConf,
    #[serde(default = "default_bandwidth")]
    pub bandwidth: BandwidthConf,
    #[serde(default = "default_acquisition")]
    pub acquisition: AcquisitionConf,
    #[serde(default = "default_sampling")]
    pub sampling: SamplingConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdvancedConf {
    #[serde(default = "default_use_restart_strategy")]
    pub use_restart_strategy: bool,
    #[serde(default = "default_restart_frequency")]
    pub restart_frequency: usize,
    #[serde(default = "default_use_adaptive_gamma")]
    pub use_adaptive_gamma: bool,
    #[serde(default = "default_use_meta_optimization")]
    pub use_meta_optimization: bool,
    #[serde(default = "default_meta_optimization_frequency")]
    pub meta_optimization_frequency: usize,
    #[serde(default = "default_use_early_stopping")]
    pub use_early_stopping: bool,
    #[serde(default = "default_early_stopping_patience")]
    pub early_stopping_patience: usize,
    #[serde(default = "default_use_constraint_aware")]
    pub use_constraint_aware: bool,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct BandwidthConf {
    #[serde(default = "default_bandwidth_method")]
    pub method: BandwidthMethod,
    #[serde(default = "default_bandwidth_cv_folds")]
    pub cv_folds: usize,
    #[serde(default = "default_bandwidth_adaptation_rate")]
    pub adaptation_rate: f64,
    #[serde(default = "default_bandwidth_min")]
    pub min_bandwidth: f64,
    #[serde(default = "default_bandwidth_max")]
    pub max_bandwidth: f64,
    #[serde(default = "default_bandwidth_cache_threshold")]
    pub cache_threshold: f64,
    #[serde(default = "default_bandwidth_min_observations")]
    pub min_observations: usize,
}

impl Default for BandwidthConf {
    fn default() -> Self {
        Self {
            method: default_bandwidth_method(),
            cv_folds: default_bandwidth_cv_folds(),
            adaptation_rate: default_bandwidth_adaptation_rate(),
            min_bandwidth: default_bandwidth_min(),
            max_bandwidth: default_bandwidth_max(),
            cache_threshold: default_bandwidth_cache_threshold(),
            min_observations: default_bandwidth_min_observations(),
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum BandwidthMethod {
    Silverman,
    CrossValidation,
    Adaptive,
    LikelihoodBased,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AcquisitionConf {
    #[serde(default = "default_acquisition_type")]
    pub acquisition_type: AcquisitionType,
    #[serde(default = "default_acquisition_xi")]
    pub xi: f64,
    #[serde(default = "default_acquisition_kappa")]
    pub kappa: f64,
    #[serde(default = "default_acquisition_use_entropy")]
    pub use_entropy: bool,
    #[serde(default = "default_acquisition_entropy_weight")]
    pub entropy_weight: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum AcquisitionType {
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityImprovement,
    EntropySearch,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SamplingConf {
    #[serde(default = "default_sampling_strategy")]
    pub strategy: SamplingStrategy,
    #[serde(default = "default_sampling_adaptive_noise")]
    pub adaptive_noise: bool,
    #[serde(default = "default_sampling_noise_scale")]
    pub noise_scale: f64,
    #[serde(default = "default_sampling_use_thompson")]
    pub use_thompson: bool,
    #[serde(default = "default_sampling_local_search")]
    pub local_search: bool,
    #[serde(default = "default_sampling_local_search_steps")]
    pub local_search_steps: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum SamplingStrategy {
    Random,
    KDEBased,
    Thompson,
    Hybrid,
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

fn default_kde_refit_frequency() -> usize {
    10
}

fn default_use_restart_strategy() -> bool {
    false
}

fn default_restart_frequency() -> usize {
    100
}

fn default_use_adaptive_gamma() -> bool {
    false
}

fn default_use_meta_optimization() -> bool {
    false
}

fn default_meta_optimization_frequency() -> usize {
    50
}

fn default_use_early_stopping() -> bool {
    false
}

fn default_early_stopping_patience() -> usize {
    50
}

fn default_use_constraint_aware() -> bool {
    false
}

fn default_advanced() -> AdvancedConf {
    AdvancedConf {
        use_restart_strategy: default_use_restart_strategy(),
        restart_frequency: default_restart_frequency(),
        use_adaptive_gamma: default_use_adaptive_gamma(),
        use_meta_optimization: default_use_meta_optimization(),
        meta_optimization_frequency: default_meta_optimization_frequency(),
        use_early_stopping: default_use_early_stopping(),
        early_stopping_patience: default_early_stopping_patience(),
        use_constraint_aware: default_use_constraint_aware(),
    }
}

fn default_bandwidth() -> BandwidthConf {
    BandwidthConf {
        method: BandwidthMethod::Silverman,
        cv_folds: 5,
        adaptation_rate: 0.1,
        min_bandwidth: 1e-6,
        max_bandwidth: 10.0,
        cache_threshold: 0.7,
        min_observations: 10,
    }
}

fn default_bandwidth_method() -> BandwidthMethod {
    BandwidthMethod::Silverman
}

fn default_bandwidth_cv_folds() -> usize {
    5
}

fn default_bandwidth_adaptation_rate() -> f64 {
    0.1
}

fn default_bandwidth_min() -> f64 {
    1e-6
}

fn default_bandwidth_max() -> f64 {
    10.0
}

fn default_bandwidth_cache_threshold() -> f64 {
    0.7
}

fn default_bandwidth_min_observations() -> usize {
    10
}

fn default_acquisition() -> AcquisitionConf {
    AcquisitionConf {
        acquisition_type: AcquisitionType::ExpectedImprovement,
        xi: 0.01,
        kappa: 2.0,
        use_entropy: false,
        entropy_weight: 0.1,
    }
}

fn default_acquisition_type() -> AcquisitionType {
    AcquisitionType::ExpectedImprovement
}

fn default_acquisition_xi() -> f64 {
    0.01
}

fn default_acquisition_kappa() -> f64 {
    2.0
}

fn default_acquisition_use_entropy() -> bool {
    false
}

fn default_acquisition_entropy_weight() -> f64 {
    0.1
}

fn default_sampling() -> SamplingConf {
    SamplingConf {
        strategy: SamplingStrategy::KDEBased,
        adaptive_noise: true,
        noise_scale: 0.1,
        use_thompson: false,
        local_search: false,
        local_search_steps: 10,
    }
}

fn default_sampling_strategy() -> SamplingStrategy {
    SamplingStrategy::KDEBased
}

fn default_sampling_adaptive_noise() -> bool {
    true
}

fn default_sampling_noise_scale() -> f64 {
    0.1
}

fn default_sampling_use_thompson() -> bool {
    false
}

fn default_sampling_local_search() -> bool {
    false
}

fn default_sampling_local_search_steps() -> usize {
    10
}

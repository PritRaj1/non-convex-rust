use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct LBFGSConf {
    pub common: CommonConf,
    pub line_search: LineSearchConf,
    pub advanced: AdvancedConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CommonConf {
    #[serde(default = "default_memory_size")]
    pub memory_size: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum LineSearchConf {
    Backtracking(BacktrackingConf),
    StrongWolfe(StrongWolfeConf),
    HagerZhang(HagerZhangConf),
    MoreThuente(MoreThuenteConf),
    GoldenSection(GoldenSectionConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct BacktrackingConf {
    #[serde(default = "default_c1")]
    pub c1: f64, // Sufficient decrease condition parameter
    #[serde(default = "default_rho")]
    pub rho: f64, // Backtracking factor
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct StrongWolfeConf {
    #[serde(default = "default_c1")]
    pub c1: f64, // Sufficient decrease condition parameter
    #[serde(default = "default_c2")]
    pub c2: f64, // Curvature condition parameter
    #[serde(default = "default_max_iters")]
    pub max_iters: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct HagerZhangConf {
    #[serde(default = "default_c1")]
    pub c1: f64, // Sufficient decrease parameter
    #[serde(default = "default_c2")]
    pub c2: f64, // Curvature condition parameter
    #[serde(default = "default_theta")]
    pub theta: f64, // Update parameter
    #[serde(default = "default_gamma")]
    pub gamma: f64, // Line search parameter
    #[serde(default = "default_max_iters")]
    pub max_iters: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MoreThuenteConf {
    #[serde(default = "default_ftol")]
    pub ftol: f64, // Function tolerance
    #[serde(default = "default_gtol")]
    pub gtol: f64, // Gradient tolerance
    #[serde(default = "default_max_iters")]
    pub max_iters: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GoldenSectionConf {
    #[serde(default = "default_tol")]
    pub tol: f64, // Tolerance for convergence
    #[serde(default = "default_max_iters")]
    pub max_iters: usize,
    #[serde(default = "default_bracket_factor")]
    pub bracket_factor: f64, // Factor for initial bracketing
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdvancedConf {
    #[serde(default = "default_adaptive_parameters")]
    pub adaptive_parameters: bool,
    #[serde(default = "default_adaptation_rate")]
    pub adaptation_rate: f64,
    #[serde(default = "default_restart_strategy")]
    pub restart_strategy: RestartStrategy,
    #[serde(default = "default_stagnation_detection")]
    pub stagnation_detection: StagnationDetection,
    #[serde(default = "default_memory_adaptation")]
    pub memory_adaptation: MemoryAdaptation,
    #[serde(default = "default_numerical_safeguards")]
    pub numerical_safeguards: NumericalSafeguards,
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
    #[serde(default = "default_gradient_threshold")]
    pub gradient_threshold: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MemoryAdaptation {
    #[serde(default = "default_adaptive_memory")]
    pub adaptive_memory: bool,
    #[serde(default = "default_min_memory_size")]
    pub min_memory_size: usize,
    #[serde(default = "default_max_memory_size")]
    pub max_memory_size: usize,
    #[serde(default = "default_memory_adaptation_rate")]
    pub memory_adaptation_rate: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NumericalSafeguards {
    #[serde(default = "default_conditioning_threshold")]
    pub conditioning_threshold: f64,
    #[serde(default = "default_curvature_threshold")]
    pub curvature_threshold: f64,
    #[serde(default = "default_use_scaling")]
    pub use_scaling: bool,
    #[serde(default = "default_scaling_factor")]
    pub scaling_factor: f64,
}

fn default_memory_size() -> usize {
    10
}
fn default_adaptive_parameters() -> bool {
    false
}
fn default_adaptation_rate() -> f64 {
    0.1
}
fn default_restart_strategy() -> RestartStrategy {
    RestartStrategy::None
}
fn default_stagnation_detection() -> StagnationDetection {
    StagnationDetection {
        stagnation_window: 50,
        improvement_threshold: 1e-6,
        gradient_threshold: 1e-6,
    }
}
fn default_memory_adaptation() -> MemoryAdaptation {
    MemoryAdaptation {
        adaptive_memory: false,
        min_memory_size: 5,
        max_memory_size: 20,
        memory_adaptation_rate: 0.1,
    }
}
fn default_numerical_safeguards() -> NumericalSafeguards {
    NumericalSafeguards {
        conditioning_threshold: 1e-12,
        curvature_threshold: 1e-8,
        use_scaling: false,
        scaling_factor: 1.0,
    }
}
fn default_success_history_size() -> usize {
    20
}
fn default_improvement_history_size() -> usize {
    20
}
fn default_stagnation_window() -> usize {
    50
}
fn default_improvement_threshold() -> f64 {
    1e-6
}
fn default_gradient_threshold() -> f64 {
    1e-6
}
fn default_adaptive_memory() -> bool {
    false
}
fn default_min_memory_size() -> usize {
    5
}
fn default_max_memory_size() -> usize {
    20
}
fn default_memory_adaptation_rate() -> f64 {
    0.1
}
fn default_conditioning_threshold() -> f64 {
    1e-12
}
fn default_curvature_threshold() -> f64 {
    1e-8
}
fn default_use_scaling() -> bool {
    false
}
fn default_scaling_factor() -> f64 {
    1.0
}
fn default_c1() -> f64 {
    0.0001
}
fn default_rho() -> f64 {
    0.5
}
fn default_c2() -> f64 {
    0.1
}
fn default_theta() -> f64 {
    0.5
}
fn default_gamma() -> f64 {
    0.5
}
fn default_max_iters() -> usize {
    100
}
fn default_ftol() -> f64 {
    1e-4
}
fn default_gtol() -> f64 {
    0.9
}
fn default_tol() -> f64 {
    1e-6
}
fn default_bracket_factor() -> f64 {
    2.0
}

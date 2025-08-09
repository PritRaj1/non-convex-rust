use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct LBFGSConf {
    pub common: CommonConf,
    pub line_search: LineSearchConf,
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

fn default_memory_size() -> usize {
    10
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

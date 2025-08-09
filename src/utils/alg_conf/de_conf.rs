use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DEConf {
    pub common: CommonConf,
    pub mutation_type: MutationType,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CommonConf {
    #[serde(default = "default_archive_size")]
    pub archive_size: usize,
    #[serde(default = "default_success_history_size")]
    pub success_history_size: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum MutationType {
    Standard(StandardConf),
    Adaptive(AdaptiveConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct StandardConf {
    #[serde(default = "default_f")]
    pub f: f64,
    #[serde(default = "default_cr")]
    pub cr: f64,
    #[serde(default = "default_strategy")]
    pub strategy: DEStrategy,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdaptiveConf {
    #[serde(default = "default_strategy")]
    pub strategy: DEStrategy,
    #[serde(default = "default_f_min")]
    pub f_min: f64,
    #[serde(default = "default_f_max")]
    pub f_max: f64,
    #[serde(default = "default_cr_min")]
    pub cr_min: f64,
    #[serde(default = "default_cr_max")]
    pub cr_max: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum DEStrategy {
    Rand1Bin,
    Best1Bin,
    RandToBest1Bin,
    Best2Bin,
    Rand2Bin,
}

fn default_archive_size() -> usize {
    10
}
fn default_f() -> f64 {
    0.8
}
fn default_cr() -> f64 {
    0.9
}
fn default_strategy() -> DEStrategy {
    DEStrategy::Rand1Bin
}
fn default_f_min() -> f64 {
    0.1
}
fn default_f_max() -> f64 {
    0.9
}
fn default_cr_min() -> f64 {
    0.1
}
fn default_cr_max() -> f64 {
    0.9
}
fn default_success_history_size() -> usize {
    50
}

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CGAConf {
    pub common: CommonConf,
    pub crossover: CrossoverConf,
    pub selection: SelectionConf,
    pub mutation: MutationConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CommonConf {
    #[serde(default = "default_num_parents")]
    pub num_parents: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum CrossoverConf {
    Random(RandomCrossoverConf),
    Heuristic(HeuristicCrossoverConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct RandomCrossoverConf {
    #[serde(default = "default_crossover_prob")]
    pub crossover_prob: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct HeuristicCrossoverConf {
    #[serde(default = "default_crossover_prob")]
    pub crossover_prob: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum SelectionConf {
    RouletteWheel(RouletteWheelSelectionConf),
    Tournament(TournamentSelectionConf),
    Residual(ResidualSelectionConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct RouletteWheelSelectionConf {}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TournamentSelectionConf {
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ResidualSelectionConf {}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum MutationConf {
    Gaussian(GaussianMutationConf),
    Uniform(UniformMutationConf),
    NonUniform(NonUniformMutationConf),
    Polynomial(PolynomialMutationConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GaussianMutationConf {
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f64,
    #[serde(default = "default_sigma")]
    pub sigma: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UniformMutationConf {
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NonUniformMutationConf {
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f64,
    #[serde(default = "default_b")]
    pub b: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PolynomialMutationConf {
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f64,
    #[serde(default = "default_eta_m")]
    pub eta_m: f64,
}

fn default_num_parents() -> usize {
    2
}
fn default_crossover_prob() -> f64 {
    0.8
}
fn default_tournament_size() -> usize {
    5
}
fn default_mutation_rate() -> f64 {
    0.01
}
fn default_sigma() -> f64 {
    0.1
}
fn default_b() -> f64 {
    5.0
}
fn default_eta_m() -> f64 {
    20.0
}

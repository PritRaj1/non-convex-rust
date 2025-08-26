use nalgebra::SMatrix;
use rand::random;
use std::hint::black_box;

use non_convex_opt::utils::config::Config;
use non_convex_opt::NonConvexOpt;

use crate::common::fcns::{Kbf, KbfConstraints, MultiModalFunction};

#[allow(dead_code)]
pub struct BenchmarkConfig {
    pub seed: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self { seed: 42 }
    }
}

#[allow(dead_code)]
pub fn generate_initial_population(_config: &BenchmarkConfig) -> SMatrix<f64, 100, 2> {
    SMatrix::<f64, 100, 2>::from_fn(|_, _| random::<f64>() * 10.0)
}

#[allow(dead_code)]
pub fn generate_initial_point(_config: &BenchmarkConfig) -> SMatrix<f64, 1, 2> {
    SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0)
}

#[allow(dead_code)]
pub fn generate_initial_simplex(_config: &BenchmarkConfig) -> SMatrix<f64, 3, 2> {
    SMatrix::<f64, 3, 2>::from_fn(|_, _| random::<f64>() * 10.0)
}

fn is_population_based(config: &Config) -> bool {
    matches!(
        &config.alg_conf,
        non_convex_opt::utils::config::AlgConf::DE(_)
            | non_convex_opt::utils::config::AlgConf::CEM(_)
            | non_convex_opt::utils::config::AlgConf::CGA(_)
            | non_convex_opt::utils::config::AlgConf::MSPO(_)
            | non_convex_opt::utils::config::AlgConf::TPE(_)
            | non_convex_opt::utils::config::AlgConf::PT(_)
    )
}

fn needs_gradient(config: &Config) -> bool {
    matches!(
        &config.alg_conf,
        non_convex_opt::utils::config::AlgConf::Adam(_)
            | non_convex_opt::utils::config::AlgConf::SGA(_)
            | non_convex_opt::utils::config::AlgConf::LBFGS(_)
    )
}

#[allow(dead_code)]
pub fn benchmark_optimization(config: &Config, bench_config: &BenchmarkConfig) {
    if is_population_based(config) {
        let init_pop = generate_initial_population(bench_config);
        let mut opt = NonConvexOpt::new(
            config.clone(),
            black_box(init_pop),
            Kbf,
            None::<KbfConstraints>,
            bench_config.seed,
        );
        let _st = opt.run();
    } else if matches!(
        &config.alg_conf,
        non_convex_opt::utils::config::AlgConf::NM(_)
    ) {
        let init_simplex = generate_initial_simplex(bench_config);
        let mut opt = NonConvexOpt::new(
            config.clone(),
            black_box(init_simplex),
            Kbf,
            None::<KbfConstraints>,
            bench_config.seed,
        );
        let _st = opt.run();
    } else {
        let init_point = generate_initial_point(bench_config);
        if needs_gradient(config) {
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_point),
                MultiModalFunction,
                None::<KbfConstraints>,
                bench_config.seed,
            );
            let _st = opt.run();
        } else {
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_point),
                Kbf,
                None::<KbfConstraints>,
                bench_config.seed,
            );
            let _st = opt.run();
        }
    }
}

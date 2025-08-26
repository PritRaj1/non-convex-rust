use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static TPE_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "TPE": {
            "n_initial_random": 20,
            "n_ei_candidates": 100,
            "gamma": 0.25,
            "prior_weight": 1.0,
            "kernel_type": "Gaussian",
            "max_history": 1000,
            "kde_refit_frequency": 5,
            "advanced": {
                "use_restart_strategy": true,
                "restart_frequency": 50,
                "use_adaptive_gamma": true,
                "use_meta_optimization": true,
                "meta_optimization_frequency": 10,
                "use_early_stopping": true,
                "early_stopping_patience": 20,
                "use_constraint_aware": true
            },
            "bandwidth": {
                "method": "Adaptive",
                "cv_folds": 5,
                "adaptation_rate": 0.1,
                "min_bandwidth": 1e-6,
                "max_bandwidth": 10.0,
                "cache_threshold": 0.2,
                "min_observations": 10
            },
            "acquisition": {
                "acquisition_type": "ExpectedImprovement",
                "xi": 0.01,
                "kappa": 2.0,
                "use_entropy": false,
                "entropy_weight": 0.1
            },
            "sampling": {
                "strategy": "Hybrid",
                "adaptive_noise": true,
                "noise_scale": 0.1,
                "use_thompson": true,
                "local_search": true,
                "local_search_steps": 10
            }
        }
    }
}"#;

static TPE_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(TPE_CONFIG_JSON).unwrap());

fn bench_tpe(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("tpe", |b| {
        b.iter(|| {
            benchmark_optimization(&TPE_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_tpe);
criterion_main!(benches);

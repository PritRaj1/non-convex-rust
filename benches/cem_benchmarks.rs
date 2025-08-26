use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static CEM_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0,
        "stagnation_window": 10
    },
    "alg_conf": {
        "CEM": {
            "common": {
                "population_size": 100,
                "elite_size": 20,
                "initial_std": 1.0,
                "min_std": 0.1,
                "max_std": 10.0
            },
            "sampling": {
                "use_antithetic": false,
                "antithetic_ratio": 0.5
            },
            "adaptation": {
                "smoothing_factor": 0.1
            },
            "advanced": {
                "use_restart_strategy": false,
                "restart_frequency": 100,
                "use_covariance_adaptation": false,
                "covariance_regularization": 1e-6,
                "improvement_history_size": 20,
                "improvement_threshold_window": 10
            }
        }
    }
}"#;

static CEM_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(CEM_CONFIG_JSON).unwrap());

fn bench_cem(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("cem", |b| {
        b.iter(|| {
            benchmark_optimization(&CEM_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_cem);
criterion_main!(benches);

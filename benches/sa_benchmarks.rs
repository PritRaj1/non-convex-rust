use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static SA_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "SA": {
            "initial_temp": 50.0,
            "cooling_rate": 0.995,
            "step_size": 0.4,
            "num_neighbors": 50,
            "x_min": 0.0,
            "x_max": 10.0,
            "min_step_size_factor": 0.3,
            "step_size_decay_power": 0.2,
            "min_temp_factor": 0.01,
            "use_adaptive_cooling": true,
            "advanced": {
                "restart_strategy": {
                    "Stagnation": {
                        "max_iterations": 100,
                        "threshold": 1e-8
                    }
                },
                "stagnation_detection": {
                    "stagnation_window": 100,
                    "improvement_threshold": 1e-8
                },
                "adaptive_parameters": true,
                "adaptation_rate": 0.15,
                "improvement_history_size": 50,
                "success_history_size": 50,
                "cooling_schedule": "Adaptive"
            }
        }
    }
}"#;

static SA_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(SA_CONFIG_JSON).unwrap());

fn bench_sa(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("sa", |b| {
        b.iter(|| {
            benchmark_optimization(&SA_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_sa);
criterion_main!(benches);

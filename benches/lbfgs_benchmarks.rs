use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static LBFGS_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "LBFGS": {
            "common": {
                "memory_size": 10
            },
            "line_search": {
                "Backtracking": {
                    "c1": 0.0001,
                    "rho": 0.5
                }
            },
            "advanced": {
                "adaptive_parameters": false,
                "adaptation_rate": 0.1,
                "restart_strategy": "None",
                "stagnation_detection": {
                    "stagnation_window": 50,
                    "improvement_threshold": 1e-6,
                    "gradient_threshold": 1e-6
                },
                "memory_adaptation": {
                    "adaptive_memory": false,
                    "min_memory_size": 5,
                    "max_memory_size": 20,
                    "memory_adaptation_rate": 0.1
                },
                "numerical_safeguards": {
                    "conditioning_threshold": 1e-12,
                    "curvature_threshold": 1e-8,
                    "use_scaling": false,
                    "scaling_factor": 1.0
                },
                "success_history_size": 20,
                "improvement_history_size": 20
            }
        }
    }
}"#;

static LBFGS_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(LBFGS_CONFIG_JSON).unwrap());

fn bench_lbfgs(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("lbfgs", |b| {
        b.iter(|| {
            benchmark_optimization(&LBFGS_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_lbfgs);
criterion_main!(benches);

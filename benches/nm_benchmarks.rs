use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static NM_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "NM": {
            "common": {
                "alpha": 1.0,
                "gamma": 2.0,
                "rho": 0.5,
                "sigma": 0.5
            },
            "advanced": {
                "adaptive_parameters": true,
                "restart_strategy": {
                    "Stagnation": {
                        "max_iterations": 30,
                        "threshold": 1e-6
                    }
                },
                "stagnation_detection": {
                    "stagnation_window": 100,
                    "improvement_threshold": 1e-6,
                    "diversity_threshold": 1e-3
                },
                "coefficient_bounds": {
                    "alpha_bounds": [0.1, 3.0],
                    "gamma_bounds": [1.0, 5.0],
                    "rho_bounds": [0.1, 1.0],
                    "sigma_bounds": [0.1, 1.0]
                },
                "adaptation_rate": 0.1,
                "success_history_size": 20,
                "improvement_history_size": 30
            }
        }
    }
}"#;

static NM_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(NM_CONFIG_JSON).unwrap());

fn bench_nm(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("nm", |b| {
        b.iter(|| {
            benchmark_optimization(&NM_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_nm);
criterion_main!(benches);

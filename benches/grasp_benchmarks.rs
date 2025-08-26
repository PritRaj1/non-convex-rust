use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static GRASP_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "GRASP": {
            "num_candidates": 100,
            "alpha": 0.5,
            "num_neighbors": 50,
            "step_size": 0.2,
            "perturbation_prob": 0.5,
            "max_local_iter": 100,
            "cache_bounds": true,
            "diversity_prob": 0.7,
            "restart_threshold": 100,
            "diversity_strength": 10.0
        }
    }
}"#;

static GRASP_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(GRASP_CONFIG_JSON).unwrap());

fn bench_grasp(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("grasp", |b| {
        b.iter(|| {
            benchmark_optimization(&GRASP_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_grasp);
criterion_main!(benches);

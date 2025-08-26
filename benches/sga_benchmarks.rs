use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static SGA_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "SGA": {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "gradient_clip": 0.0,
            "noise_decay": 0.99,
            "adaptive_noise": false
        }
    }
}"#;

static SGA_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(SGA_CONFIG_JSON).unwrap());

fn bench_sga(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("sga", |b| {
        b.iter(|| {
            benchmark_optimization(&SGA_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_sga);
criterion_main!(benches);

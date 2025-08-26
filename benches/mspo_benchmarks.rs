use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static MSPO_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "MSPO": {
            "num_swarms": 10,
            "swarm_size": 10,
            "c1": 1.5,
            "c2": 1.5,
            "x_min": 0.0,
            "x_max": 10.0,
            "exchange_interval": 20,
            "exchange_ratio": 0.05,
            "improvement_threshold": 0.1,
            "inertia_start": 0.9,
            "inertia_end": 0.7
        }
    }
}"#;

static MSPO_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(MSPO_CONFIG_JSON).unwrap());

fn bench_mspo(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("mspo", |b| {
        b.iter(|| {
            benchmark_optimization(&MSPO_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_mspo);
criterion_main!(benches);

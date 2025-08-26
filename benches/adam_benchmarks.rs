use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static ADAM_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "Adam": {
            "learning_rate": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip": 0.0,
            "amsgrad": false
        }
    }
}"#;

static ADAM_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(ADAM_CONFIG_JSON).unwrap());

fn bench_adam(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("adam", |b| {
        b.iter(|| {
            benchmark_optimization(&ADAM_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_adam);
criterion_main!(benches);

use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static CMAES_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "CMAES": {
            "num_parents": 50,
            "initial_sigma": 1.5
        }
    }
}"#;

static CMAES_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(CMAES_CONFIG_JSON).unwrap());

fn bench_cmaes(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("cmaes", |b| {
        b.iter(|| {
            benchmark_optimization(&CMAES_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_cmaes);
criterion_main!(benches);

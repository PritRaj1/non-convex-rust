use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static CGA_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0,
        "stagnation_window": 10
    },
    "alg_conf": {
        "CGA": {
            "common": {
                "num_parents": 100
            },
            "crossover": {
                "Heuristic": {
                    "crossover_prob": 0.8
                }
            },
            "selection": {
                "Tournament": {
                    "tournament_size": 50
                }
            },
            "mutation": {
                "Gaussian": {
                    "mutation_rate": 0.1,
                    "sigma": 1.0
                }
            }
        }
    }
}"#;

static CGA_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(CGA_CONFIG_JSON).unwrap());

fn bench_cga(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("cga", |b| {
        b.iter(|| {
            benchmark_optimization(&CGA_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_cga);
criterion_main!(benches);

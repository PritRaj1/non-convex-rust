use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static DE_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "DE": {
            "common": {
                "archive_size": 10,
                "success_history_size": 50
            },
            "mutation_type": {
                "Adaptive": {
                    "strategy": "Best2Bin",
                    "f_min": 0.4,
                    "f_max": 0.9,
                    "cr_min": 0.1,
                    "cr_max": 0.9,
                    "use_jade": false,
                    "memory_size": 5,
                    "learning_rate": 0.1
                }
            }
        }
    }
}"#;

static DE_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(DE_CONFIG_JSON).unwrap());

fn bench_de(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("de", |b| {
        b.iter(|| {
            benchmark_optimization(&DE_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_de);
criterion_main!(benches);

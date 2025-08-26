use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static PT_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": "0.0",
        "atol": "0.0",
        "rtol_max_iter_fraction": 1.0,
        "stagnation_window": 10
    },
    "alg_conf": {
        "PT": {
            "common": {
                "num_replicas": 5,
                "power_law_init": 2.0,
                "power_law_final": 0.5,
                "power_law_cycles": 1,
                "alpha": 0.1,
                "omega": 2.1,
                "mala_step_size": 0.1
            },
            "swap_conf": {
                "Always": {}
            },
            "update_conf": {
                "Auto": {}
            }
        }
    }
}"#;

static PT_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(PT_CONFIG_JSON).unwrap());

fn bench_pt(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("pt", |b| {
        b.iter(|| {
            benchmark_optimization(&PT_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_pt);
criterion_main!(benches);

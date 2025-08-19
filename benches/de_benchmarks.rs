use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;
use std::hint::black_box;
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;
use non_convex_opt::NonConvexOpt;

mod common;
use common::fcns::{KbfConstraints, Kbf};

static CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
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
                    "cr_max": 0.9
                }
            }
        }
    }
}"#;

static CONFIG: LazyLock<Config> = LazyLock::new(|| serde_json::from_str(CONFIG_JSON).unwrap());

fn bench_de_unconstrained(c: &mut Criterion) {
    c.bench_function("de_unconstrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 100, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(init_pop),
                Kbf,
                None::<KbfConstraints>,
            );
            let _st = opt.run();
        })
    });
}

fn bench_de_constrained(c: &mut Criterion) {
    c.bench_function("de_constrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 100, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(init_pop),
                Kbf,
                Some(KbfConstraints),
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_de_unconstrained, bench_de_constrained);
criterion_main!(benches);

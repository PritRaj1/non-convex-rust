use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;
use std::hint::black_box;
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;
use non_convex_opt::NonConvexOpt;

mod common;
use common::fcns::{Kbf, KbfConstraints};

static CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "MSPO": {
            "num_swarms": 10,
            "swarm_size": 10,
            "w": 0.729,
            "c1": 1.5,
            "c2": 1.5,
            "x_min": 0.0,
            "x_max": 10.0,
            "exchange_interval": 20,
            "exchange_ratio": 0.05
        }
    }
}"#;

static CONFIG: LazyLock<Config> = LazyLock::new(|| serde_json::from_str(CONFIG_JSON).unwrap());

fn bench_mspo_unconstrained(c: &mut Criterion) {
    c.bench_function("mspo_unconstrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 100, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(init_pop),
                Kbf,
                None::<KbfConstraints>,
                42,
            );
            let _st = opt.run();
        })
    });
}

fn bench_mspo_constrained(c: &mut Criterion) {
    c.bench_function("mspo_constrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 100, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(init_pop),
                Kbf,
                Some(KbfConstraints),
                42,
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_mspo_unconstrained, bench_mspo_constrained);
criterion_main!(benches);

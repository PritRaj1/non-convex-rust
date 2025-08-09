use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;
use non_convex_opt::NonConvexOpt;

mod common;
use common::fcns::{KBFConstraints, KBF};

static CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "PT": {
            "common": {
                "num_replicas": 10,
                "power_law_init": 2.0,
                "power_law_final": 0.5,
                "power_law_cycles": 1,
                "alpha": 0.1,
                "omega": 2.1,
                "mala_step_size": 0.1
            },
            "swap_conf": {
                "Always": {}
            }
        }
    }
}"#;

static CONFIG: LazyLock<Config> = LazyLock::new(|| serde_json::from_str(CONFIG_JSON).unwrap());

fn bench_pt_unconstrained(c: &mut Criterion) {
    c.bench_function("pt_unconstrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 100, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(init_pop),
                KBF,
                None::<KBFConstraints>,
            );
            let _st = opt.run();
        })
    });
}

fn bench_pt_constrained(c: &mut Criterion) {
    c.bench_function("pt_constrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 100, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(init_pop),
                KBF,
                Some(KBFConstraints),
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_pt_unconstrained, bench_pt_constrained);
criterion_main!(benches);

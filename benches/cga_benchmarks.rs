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
        "CGA": {
            "common": {
                "num_parents": 20
            },
            "crossover": {
                "Heuristic": {
                    "crossover_prob": 0.5
                }
            },
            "selection": {
                "Tournament": {
                    "tournament_size": 50
                }
            },
            "mutation": {
                "NonUniform": {
                    "mutation_rate": 0.1,
                    "b": 5.0
                }
            }
        }
    }
}"#;

static CONFIG: LazyLock<Config> = LazyLock::new(|| serde_json::from_str(CONFIG_JSON).unwrap());

fn bench_cga_unconstrained(c: &mut Criterion) {
    c.bench_function("cga_unconstrained", |b| {
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

fn bench_cga_constrained(c: &mut Criterion) {
    c.bench_function("cga_constrained", |b| {
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

criterion_group!(benches, bench_cga_unconstrained, bench_cga_constrained);
criterion_main!(benches);

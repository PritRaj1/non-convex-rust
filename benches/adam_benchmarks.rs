use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;
use std::hint::black_box;

use non_convex_opt::utils::config::{AdamConf, AlgConf, Config, OptConf};
use non_convex_opt::NonConvexOpt;

mod common;
use common::fcns::{RosenbrockConstraints, RosenbrockFunction};

fn bench_adam_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::Adam(AdamConf {
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            gradient_clip: 1.0,
            amsgrad: false,
        }),
    };

    c.bench_function("adam_unconstrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                None::<RosenbrockConstraints>,
                42,
            );
            let _st = opt.run();
        })
    });
}

fn bench_adam_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::Adam(AdamConf {
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            gradient_clip: 1.0,
            amsgrad: false,
        }),
    };

    c.bench_function("adam_constrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                Some(RosenbrockConstraints),
                42,
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_adam_unconstrained, bench_adam_constrained);
criterion_main!(benches);

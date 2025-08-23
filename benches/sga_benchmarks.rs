use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;
use std::hint::black_box;

use non_convex_opt::utils::config::{AlgConf, Config, OptConf, SGAConf};
use non_convex_opt::NonConvexOpt;

mod common;
use common::fcns::{RosenbrockConstraints, RosenbrockFunction};

fn bench_sga_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::SGA(SGAConf {
            learning_rate: 0.01,
            momentum: 0.9,
            gradient_clip: 1.0,
            noise_decay: 0.99,
            adaptive_noise: false,
        }),
    };

    c.bench_function("sga_unconstrained", |b| {
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

criterion_group!(benches, bench_sga_unconstrained);
criterion_main!(benches);

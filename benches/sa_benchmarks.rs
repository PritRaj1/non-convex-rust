use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;

use non_convex_opt::utils::config::{AlgConf, Config, OptConf, SAConf};
use non_convex_opt::NonConvexOpt;

mod common;
use common::fcns::{KBFConstraints, KBF};

fn bench_sa_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::SA(SAConf {
            initial_temp: 1000.0,
            cooling_rate: 0.998,
            step_size: 0.5,
            num_neighbors: 20,
            reheat_after: 50,
            x_min: 0.0,
            x_max: 10.0,
        }),
    };

    c.bench_function("sa_unconstrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                KBF,
                None::<KBFConstraints>,
            );
            let _st = opt.run();
        })
    });
}

fn bench_sa_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::SA(SAConf {
            initial_temp: 1000.0,
            cooling_rate: 0.998,
            step_size: 0.5,
            num_neighbors: 20,
            reheat_after: 50,
            x_min: 0.0,
            x_max: 10.0,
        }),
    };

    c.bench_function("sa_constrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                KBF,
                Some(KBFConstraints),
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_sa_unconstrained, bench_sa_constrained);
criterion_main!(benches);

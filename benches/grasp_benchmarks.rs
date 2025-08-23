use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;
use std::hint::black_box;

use non_convex_opt::utils::config::{AlgConf, Config, GRASPConf, OptConf};
use non_convex_opt::NonConvexOpt;

mod common;
use common::fcns::{Kbf, KbfConstraints};

fn bench_grasp_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::GRASP(GRASPConf {
            num_candidates: 100,
            alpha: 0.5,
            num_neighbors: 50,
            step_size: 0.2,
            perturbation_prob: 0.5,
            max_local_iter: 100,
            cache_bounds: true,
            diversity_prob: 0.7,
            restart_threshold: 15,
            diversity_strength: 10.0,
        }),
    };

    c.bench_function("grasp_unconstrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                Kbf,
                None::<KbfConstraints>,
                42,
            );
            let _st = opt.run();
        })
    });
}

fn bench_grasp_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::GRASP(GRASPConf {
            num_candidates: 100,
            alpha: 0.5,
            num_neighbors: 50,
            step_size: 0.2,
            perturbation_prob: 0.5,
            max_local_iter: 100,
            cache_bounds: true,
            diversity_prob: 0.7,
            restart_threshold: 15,
            diversity_strength: 10.0,
        }),
    };

    c.bench_function("grasp_constrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                Kbf,
                Some(KbfConstraints),
                42,
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_grasp_unconstrained, bench_grasp_constrained);
criterion_main!(benches);

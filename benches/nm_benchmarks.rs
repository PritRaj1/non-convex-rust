mod common;
use common::fcns::{KBFConstraints, KBF};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{SMatrix, SVector};

use non_convex_opt::utils::config::{AlgConf, Config, NelderMeadConf, OptConf};
use non_convex_opt::NonConvexOpt;

fn bench_nm_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::NM(NelderMeadConf {
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }),
    };

    c.bench_function("nm_unconstrained", |b| {
        b.iter(|| {
            let init_simplex = SMatrix::<f64, 3, 2>::from_rows(&[
                SMatrix::<f64, 1, 2>::from_row_slice(&[1.8, 1.0]),
                SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 4.0]),
                SMatrix::<f64, 1, 2>::from_row_slice(&[3.0, 3.0]),
            ]);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_simplex),
                KBF,
                None::<KBFConstraints>,
            );
            let _st = opt.run();
        })
    });
}

fn bench_nm_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::NM(NelderMeadConf {
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }),
    };

    c.bench_function("nm_constrained", |b| {
        b.iter(|| {
            let init_simplex = SMatrix::<f64, 3, 2>::from_columns(&[
                SVector::<f64, 3>::from_vec(vec![1.8, 1.0, 0.0]),
                SVector::<f64, 3>::from_vec(vec![0.5, 4.0, 0.0]),
            ]);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_simplex),
                KBF,
                Some(KBFConstraints),
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_nm_unconstrained, bench_nm_constrained);
criterion_main!(benches);

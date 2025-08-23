use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{SMatrix, SVector};
use std::hint::black_box;
use std::sync::LazyLock;

use common::fcns::{Kbf, KbfConstraints};
use non_convex_opt::utils::alg_conf::nm_conf::{
    AdvancedConf, CoefficientBounds, RestartStrategy, StagnationDetection,
};
use non_convex_opt::utils::config::{AlgConf, Config, NelderMeadConf, OptConf};
use non_convex_opt::NonConvexOpt;

mod common;

#[allow(dead_code)]
static CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 100,
        "rtol": 1e-6,
        "atol": 0.0,
        "rtol_max_iter_fraction": 1.0,
        "stagnation_window": 50
    },
    "alg_conf": {
        "NM": {
            "common": {
                "alpha": 1.0,
                "gamma": 2.0,
                "rho": 0.5,
                "sigma": 0.5
            },
            "advanced": {
                "adaptive_parameters": true,
                "restart_strategy": {
                    "Stagnation": {
                        "max_iterations": 30,
                        "threshold": 1e-6
                    }
                },
                "stagnation_detection": {
                    "stagnation_window": 20,
                    "improvement_threshold": 1e-6,
                    "diversity_threshold": 1e-3
                },
                "coefficient_bounds": {
                    "alpha_bounds": [0.1, 3.0],
                    "gamma_bounds": [1.0, 5.0],
                    "rho_bounds": [0.1, 1.0],
                    "sigma_bounds": [0.1, 1.0]
                },
                "adaptation_rate": 0.1,
                "success_history_size": 20,
                "improvement_history_size": 30
            }
        }
    }
}
"#;

#[allow(dead_code)]
static CONFIG: LazyLock<Config> = LazyLock::new(|| serde_json::from_str(CONFIG_JSON).unwrap());

fn bench_nm_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 100,
            rtol: 1e-6,
            atol: 0.0,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::NM(NelderMeadConf {
            common: non_convex_opt::utils::alg_conf::nm_conf::CommonConf {
                alpha: 1.0,
                gamma: 2.0,
                rho: 0.5,
                sigma: 0.5,
            },
            advanced: AdvancedConf {
                adaptive_parameters: true,
                restart_strategy: RestartStrategy::Stagnation {
                    max_iterations: 30,
                    threshold: 1e-6,
                },
                stagnation_detection: StagnationDetection {
                    stagnation_window: 20,
                    improvement_threshold: 1e-6,
                    diversity_threshold: 1e-3,
                },
                coefficient_bounds: CoefficientBounds {
                    alpha_bounds: (0.1, 3.0),
                    gamma_bounds: (1.0, 5.0),
                    rho_bounds: (0.1, 1.0),
                    sigma_bounds: (0.1, 1.0),
                },
                adaptation_rate: 0.1,
                success_history_size: 20,
                improvement_history_size: 30,
            },
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
                Kbf,
                None::<KbfConstraints>,
                42,
            );
            let _st = opt.run();
        })
    });
}

fn bench_nm_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 100,
            rtol: 1e-6,
            atol: 0.0,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::NM(NelderMeadConf {
            common: non_convex_opt::utils::alg_conf::nm_conf::CommonConf {
                alpha: 1.0,
                gamma: 2.0,
                rho: 0.5,
                sigma: 0.5,
            },
            advanced: AdvancedConf {
                adaptive_parameters: true,
                restart_strategy: RestartStrategy::Stagnation {
                    max_iterations: 30,
                    threshold: 1e-6,
                },
                stagnation_detection: StagnationDetection {
                    stagnation_window: 20,
                    improvement_threshold: 1e-6,
                    diversity_threshold: 1e-3,
                },
                coefficient_bounds: CoefficientBounds {
                    alpha_bounds: (0.1, 3.0),
                    gamma_bounds: (1.0, 5.0),
                    rho_bounds: (0.1, 1.0),
                    sigma_bounds: (0.1, 1.0),
                },
                adaptation_rate: 0.1,
                success_history_size: 20,
                improvement_history_size: 30,
            },
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
                Kbf,
                Some(KbfConstraints),
                42,
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_nm_unconstrained, bench_nm_constrained);
criterion_main!(benches);

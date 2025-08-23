use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;
use std::hint::black_box;
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;
use non_convex_opt::NonConvexOpt;

mod common;
use common::fcns::{RosenbrockConstraints, RosenbrockFunction};

static BACKTRACKING_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "LBFGS": {
            "common": {
                "memory_size": 10
            },
            "line_search": {
                "Backtracking": {
                    "c1": 0.0001,
                    "c2": 0.9,
                    "max_iters": 100
                }
            }
        }
    }
}
"#;

static HAGER_ZHANG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "LBFGS": {
            "common": {
                "memory_size": 10
            },
            "line_search": {
                "HagerZhang": {
                    "c1": 0.0001,
                    "c2": 0.9,
                    "theta": 0.5,
                    "gamma": 0.5,
                    "max_iters": 100
                }
            }
        }
    }   
}
"#;

static MORE_THUENTE_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "LBFGS": {
            "common": {
                "memory_size": 10
            },
            "line_search": {
                "MoreThuente": {
                    "ftol": 1e-4,
                    "gtol": 0.9,
                    "max_iters": 100
                }
            }       
        }
    }
}
"#;

static GOLDEN_SECTION_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "LBFGS": {
            "common": {
                "memory_size": 10
            },
            "line_search": {
                "GoldenSection": {
                    "tol": 1e-6,
                    "max_iters": 100,
                    "bracket_factor": 2.0
                }
            }
        }
    }
}
"#;

static STRONG_WOLFE_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "LBFGS": {
            "common": {
                "memory_size": 10
            },
            "line_search": {
                "StrongWolfe": {
                    "c1": 0.0001,
                    "c2": 0.9,
                    "max_iters": 100
                }
            }
        }
    }
}
"#;

static BACKTRACKING_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(BACKTRACKING_JSON).unwrap());

static HAGER_ZHANG_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(HAGER_ZHANG_JSON).unwrap());

static MORE_THUENTE_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(MORE_THUENTE_JSON).unwrap());

static GOLDEN_SECTION_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(GOLDEN_SECTION_JSON).unwrap());

static STRONG_WOLFE_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(STRONG_WOLFE_JSON).unwrap());

fn bench_lbfgs_backtracking(c: &mut Criterion) {
    c.bench_function("lbfgs_backtracking", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                BACKTRACKING_CONFIG.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                None::<RosenbrockConstraints>,
                42,
            );
            let _st = opt.run();
        })
    });
}

fn bench_lbfgs_hager_zhang(c: &mut Criterion) {
    c.bench_function("lbfgs_hager_zhang", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                HAGER_ZHANG_CONFIG.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                None::<RosenbrockConstraints>,
                42,
            );
            let _st = opt.run();
        })
    });
}

fn bench_lbfgs_more_thuente(c: &mut Criterion) {
    c.bench_function("lbfgs_more_thuente", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                MORE_THUENTE_CONFIG.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                None::<RosenbrockConstraints>,
                42,
            );
            let _st = opt.run();
        })
    });
}

fn bench_lbfgs_golden_section(c: &mut Criterion) {
    c.bench_function("lbfgs_golden_section", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                GOLDEN_SECTION_CONFIG.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                None::<RosenbrockConstraints>,
                42,
            );
            let _st = opt.run();
        })
    });
}

fn bench_lbfgs_strong_wolfe(c: &mut Criterion) {
    c.bench_function("lbfgs_strong_wolfe", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                STRONG_WOLFE_CONFIG.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                None::<RosenbrockConstraints>,
                42,
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(
    benches,
    bench_lbfgs_backtracking,
    bench_lbfgs_hager_zhang,
    bench_lbfgs_more_thuente,
    bench_lbfgs_golden_section,
    bench_lbfgs_strong_wolfe
);

criterion_main!(benches);

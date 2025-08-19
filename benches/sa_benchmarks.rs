use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;
use std::hint::black_box;

use non_convex_opt::utils::alg_conf::sa_conf::{
    AdvancedConf, CoolingScheduleType, RestartStrategy, SAConf, StagnationDetection,
};
use non_convex_opt::utils::config::{AlgConf, Config, OptConf};
use non_convex_opt::NonConvexOpt;

mod common;
use common::fcns::{Kbf, KbfConstraints};

fn bench_sa_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
            stagnation_window: 50,
        },
        alg_conf: AlgConf::SA(SAConf {
            initial_temp: 1000.0,
            cooling_rate: 0.998,
            step_size: 1.0,
            num_neighbors: 20,
            x_min: 0.0,
            x_max: 10.0,
            min_step_size_factor: 0.3,
            step_size_decay_power: 0.2,
            min_temp_factor: 0.05,
            use_adaptive_cooling: false,
            advanced: AdvancedConf {
                restart_strategy: RestartStrategy::None,
                stagnation_detection: StagnationDetection {
                    stagnation_window: 10,
                    improvement_threshold: 1e-6,
                },
                adaptive_parameters: false,
                adaptation_rate: 0.1,
                improvement_history_size: 20,
                success_history_size: 20,
                cooling_schedule: CoolingScheduleType::Exponential,
            },
        }),
    };

    c.bench_function("sa_unconstrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                Kbf,
                None::<KbfConstraints>,
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
            stagnation_window: 50,
        },
        alg_conf: AlgConf::SA(SAConf {
            initial_temp: 1000.0,
            cooling_rate: 0.998,
            step_size: 1.0,
            num_neighbors: 20,
            x_min: 0.0,
            x_max: 10.0,
            min_step_size_factor: 0.3,
            step_size_decay_power: 0.2,
            min_temp_factor: 0.05,
            use_adaptive_cooling: false,
            advanced: AdvancedConf {
                restart_strategy: RestartStrategy::None,
                stagnation_detection: StagnationDetection {
                    stagnation_window: 10,
                    improvement_threshold: 1e-6,
                },
                adaptive_parameters: false,
                adaptation_rate: 0.1,
                improvement_history_size: 20,
                success_history_size: 20,
                cooling_schedule: CoolingScheduleType::Exponential,
            },
        }),
    };

    c.bench_function("sa_constrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                Kbf,
                Some(KbfConstraints),
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_sa_unconstrained, bench_sa_constrained);
criterion_main!(benches);

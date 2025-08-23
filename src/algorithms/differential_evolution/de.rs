use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::utils::alg_conf::de_conf::{DEConf, DEStrategy, MutationType};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::differential_evolution::{
    archive::Archive,
    bounds::BoundsCache,
    mutation::{Best1Bin, Best2Bin, MutationStrategy, Rand1Bin, Rand2Bin, RandToBest1Bin},
    parameter_adaptation::{
        JADEParameterAdaptation, ParameterAdaptationType, StandardParameterAdaptation,
    },
};

pub struct DE<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    pub conf: DEConf,
    pub st: State<T, N, D>,
    pub opt_prob: OptProb<T, D>,
    pub archive: Archive<T, D>,
    success_history: VecDeque<bool>,
    parameter_adaptation: ParameterAdaptationType,
    bounds_cache: BoundsCache<T, D>,
    seed: u64,
}

impl<T, N, D> DE<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    pub fn new(
        conf: DEConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        seed: u64,
    ) -> Self {
        let population_size = init_pop.nrows();
        let mut fitness = OVector::<T, N>::zeros_generic(N::from_usize(population_size), U1);
        let mut constraints =
            OVector::<bool, N>::from_element_generic(N::from_usize(population_size), U1, true);

        let evaluations: Vec<(T, bool)> = (0..population_size)
            .into_par_iter()
            .map(|i| {
                let x = init_pop.row(i).transpose();
                let fit = opt_prob.evaluate(&x);
                let constr = opt_prob.is_feasible(&x);
                (fit, constr)
            })
            .collect();

        for (i, (fit, constr)) in evaluations.into_iter().enumerate() {
            fitness[i] = fit;
            constraints[i] = constr;
        }

        let mut best_idx = 0;
        let mut best_fitness = fitness[0];
        for i in 1..population_size {
            if fitness[i] > best_fitness && constraints[i] {
                best_idx = i;
                best_fitness = fitness[i];
            }
        }

        let archive_size = conf.common.archive_size;
        let success_history_size = conf.common.success_history_size;

        let parameter_adaptation = match &conf.mutation_type {
            MutationType::Standard(standard) => {
                ParameterAdaptationType::Standard(StandardParameterAdaptation::new(
                    standard.f,
                    standard.cr,
                    standard.f * 0.5,  // f_min
                    standard.f * 1.5,  // f_max
                    standard.cr * 0.5, // cr_min
                    standard.cr * 1.5, // cr_max
                ))
            }
            MutationType::Adaptive(adaptive) => {
                if adaptive.use_jade {
                    ParameterAdaptationType::JADE(Box::new(JADEParameterAdaptation::new(
                        adaptive.memory_size,
                        seed,
                    )))
                } else {
                    ParameterAdaptationType::Standard(StandardParameterAdaptation::new(
                        (adaptive.f_min + adaptive.f_max) / 2.0,
                        (adaptive.cr_min + adaptive.cr_max) / 2.0,
                        adaptive.f_min,
                        adaptive.f_max,
                        adaptive.cr_min,
                        adaptive.cr_max,
                    ))
                }
            }
        };

        Self {
            conf,
            st: State {
                pop: init_pop.clone(),
                fitness,
                constraints,
                best_x: init_pop.row(best_idx).transpose(),
                best_f: best_fitness,
                iter: 1,
            },
            opt_prob,
            archive: Archive::new(archive_size, seed),
            success_history: VecDeque::with_capacity(success_history_size),
            parameter_adaptation,
            bounds_cache: BoundsCache::new(init_pop.ncols()),
            seed,
        }
    }

    fn select_trial(
        &self,
        trial_fitness: T,
        trial_constraint: bool,
        current_fitness: T,
        current_constraint: bool,
    ) -> bool {
        match (trial_constraint, current_constraint) {
            (true, true) => {
                // Both feasible - compare fitness with tolerance
                let eps = T::from_f64(1e-10).unwrap();
                trial_fitness > current_fitness + eps
            }
            (true, false) => true,  // Prefer feasible
            (false, true) => false, // Keep feasible
            (false, false) => {
                // Both infeasible - compare fitness
                trial_fitness > current_fitness
            }
        }
    }
}

impl<T: FloatNum, N: Dim, D: Dim> OptimizationAlgorithm<T, N, D> for DE<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        let pop_size = self.st.pop.nrows();

        let sample_individual = self.st.pop.row(0).transpose();
        let _bounds = self
            .bounds_cache
            .get_bounds(&self.opt_prob, &sample_individual);

        let trials: Vec<_> = (0..pop_size)
            .into_par_iter()
            .map_init(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    StdRng::seed_from_u64(self.seed + self.st.iter as u64 * 1000 + thread_id as u64)
                },
                |rng, i| {
                    let strategy = match &self.conf.mutation_type {
                        MutationType::Standard(standard) => &standard.strategy,
                        MutationType::Adaptive(adaptive) => &adaptive.strategy,
                    };

                    let strategy: &dyn MutationStrategy<T, N, D> = match strategy {
                        DEStrategy::Rand1Bin => &Rand1Bin,
                        DEStrategy::Best1Bin => &Best1Bin,
                        DEStrategy::RandToBest1Bin => &RandToBest1Bin,
                        DEStrategy::Best2Bin => &Best2Bin,
                        DEStrategy::Rand2Bin => &Rand2Bin,
                    };

                    let mut local_pa = self.parameter_adaptation.clone();
                    let (f, cr) = local_pa.generate_parameters();

                    let trial = strategy.generate_trial(
                        &self.st.pop,
                        Some(&self.st.best_x),
                        i,
                        T::from_f64(f).unwrap(),
                        T::from_f64(cr).unwrap(),
                        rng,
                    );

                    let fitness = self.opt_prob.evaluate(&trial);
                    let constraint = self.opt_prob.is_feasible(&trial);

                    let success = self.select_trial(
                        fitness,
                        constraint,
                        self.st.fitness[i],
                        self.st.constraints[i],
                    );

                    (i, trial, fitness, constraint, success, f, cr)
                },
            )
            .collect();

        let mut successes = Vec::new();

        let updates: Vec<_> = trials
            .into_iter()
            .filter_map(
                |(i, trial, trial_fitness, trial_constraint, success, f, cr)| {
                    if trial_constraint && trial_fitness > self.st.fitness[i] {
                        self.archive.add_solution(trial.clone(), trial_fitness);
                        self.parameter_adaptation.record_success(f, cr);
                    }

                    successes.push(success);

                    if success {
                        Some((i, trial, trial_fitness, trial_constraint))
                    } else {
                        None
                    }
                },
            )
            .collect();

        for success in successes {
            self.success_history.push_back(success);
            if self.success_history.len() > self.conf.common.success_history_size {
                self.success_history.pop_front();
            }
        }

        self.parameter_adaptation.update_parameters();

        let mut new_population = self.st.pop.clone();
        let mut new_fitness = self.st.fitness.clone();
        let mut new_constraints = self.st.constraints.clone();

        for (i, trial, trial_fitness, trial_constraint) in updates {
            new_population.set_row(i, &trial.transpose());
            new_fitness[i] = trial_fitness;
            new_constraints[i] = trial_constraint;
        }

        self.st.pop = new_population;
        self.st.fitness = new_fitness;
        self.st.constraints = new_constraints;

        for i in 0..pop_size {
            if self.st.constraints[i] && self.st.fitness[i] > self.st.best_f {
                self.st.best_f = self.st.fitness[i];
                self.st.best_x = self.st.pop.row(i).transpose();
            }
        }

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

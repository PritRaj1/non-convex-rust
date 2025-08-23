use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1};
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::utils::config::{CGAConf, CrossoverConf, MutationConf, SelectionConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::continuous_genetic::{
    crossover::*,
    mutation::{Gaussian, MutationOperator, MutationOperatorEnum, NonUniform, Polynomial, Uniform},
    selection::*,
};

pub struct CGA<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: CGAConf,
    pub st: State<T, N, D>,
    pub opt_prob: OptProb<T, D>,
    pub selector: Box<dyn SelectionOperator<T, N, D> + Send + Sync>,
    pub crossover: Box<dyn CrossoverOperator<T, N, D> + Send + Sync>,
    pub mutation: MutationOperatorEnum<T, D>,
    cached_bounds_lower: OVector<T, D>,
    cached_bounds_upper: OVector<T, D>,
    bounds_cached: bool,
    success_history: VecDeque<bool>,
    current_mutation_rate: f64,
    current_crossover_prob: f64,
    generation_improvements: VecDeque<f64>,
}

impl<T, N, D> CGA<T, N, D>
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D>,
{
    pub fn new(
        conf: CGAConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        max_iter: usize,
        seed: u64,
    ) -> Self {
        let selector: Box<dyn SelectionOperator<T, N, D> + Send + Sync> = match &conf.selection {
            SelectionConf::RouletteWheel(_) => Box::new(RouletteWheel::new(
                init_pop.nrows(),
                conf.common.num_parents,
                seed,
            )),
            SelectionConf::Tournament(tournament) => Box::new(Tournament::new(
                init_pop.nrows(),
                conf.common.num_parents,
                tournament.tournament_size,
                seed,
            )),
            SelectionConf::Residual(_) => Box::new(Residual::new(
                init_pop.nrows(),
                conf.common.num_parents,
                seed,
            )),
        };

        let crossover: Box<dyn CrossoverOperator<T, N, D> + Send + Sync> = match &conf.crossover {
            CrossoverConf::Random(random) => {
                Box::new(Random::new(random.crossover_prob, init_pop.nrows(), seed))
            }
            CrossoverConf::Heuristic(heuristic) => Box::new(Heuristic::new(
                heuristic.crossover_prob,
                init_pop.nrows(),
                seed,
            )),
            CrossoverConf::SimulatedBinary(sbx) => Box::new(SimulatedBinary::new(
                sbx.crossover_prob,
                sbx.eta_c,
                init_pop.nrows(),
                seed,
            )),
        };

        let mutation: MutationOperatorEnum<T, D> = match &conf.mutation {
            MutationConf::Gaussian(gaussian) => MutationOperatorEnum::<T, D>::Gaussian(
                Gaussian::new(gaussian.mutation_rate, gaussian.sigma, seed),
                std::marker::PhantomData,
            ),
            MutationConf::Uniform(uniform) => MutationOperatorEnum::<T, D>::Uniform(
                Uniform::new(uniform.mutation_rate, seed),
                std::marker::PhantomData,
            ),
            MutationConf::NonUniform(non_uniform) => MutationOperatorEnum::<T, D>::NonUniform(
                NonUniform::new(non_uniform.mutation_rate, non_uniform.b, max_iter, seed),
                std::marker::PhantomData,
            ),
            MutationConf::Polynomial(polynomial) => MutationOperatorEnum::<T, D>::Polynomial(
                Polynomial::new(polynomial.mutation_rate, polynomial.eta_m, seed),
                std::marker::PhantomData,
            ),
        };

        // Calculate initial fitness and constraints in parallel
        let (fitness, constraints): (Vec<T>, Vec<bool>) = (0..init_pop.nrows())
            .into_par_iter()
            .map(|i| {
                let individual = init_pop.row(i).transpose();
                let fit = opt_prob.evaluate(&individual);
                let constr = opt_prob.is_feasible(&individual);
                (fit, constr)
            })
            .unzip();

        let fitness =
            OVector::<T, N>::from_vec_generic(N::from_usize(init_pop.nrows()), U1, fitness);
        let constraints =
            OVector::<bool, N>::from_vec_generic(N::from_usize(init_pop.nrows()), U1, constraints);

        // Find best individual
        let mut best_idx = 0;
        let mut best_fitness = fitness[0];
        for i in 1..fitness.len() {
            if fitness[i] > best_fitness && constraints[i] {
                best_idx = i;
                best_fitness = fitness[i];
            }
        }
        let best_individual = init_pop.row(best_idx).transpose();

        let initial_mutation_rate = match &conf.mutation {
            MutationConf::Gaussian(g) => g.mutation_rate,
            MutationConf::Uniform(u) => u.mutation_rate,
            MutationConf::NonUniform(nu) => nu.mutation_rate,
            MutationConf::Polynomial(p) => p.mutation_rate,
        };

        let initial_crossover_prob = match &conf.crossover {
            CrossoverConf::Random(r) => r.crossover_prob,
            CrossoverConf::Heuristic(h) => h.crossover_prob,
            CrossoverConf::SimulatedBinary(sbx) => sbx.crossover_prob,
        };

        let success_history_size = conf.common.success_history_size;

        let n = init_pop.ncols();
        Self {
            conf,
            st: State {
                pop: init_pop,
                fitness,
                constraints,
                best_x: best_individual,
                best_f: best_fitness,
                iter: 1,
            },
            opt_prob,
            selector,
            crossover,
            mutation,
            cached_bounds_lower: OVector::<T, D>::from_element_generic(
                D::from_usize(n),
                U1,
                T::from_f64(-10.0).unwrap(),
            ),
            cached_bounds_upper: OVector::<T, D>::from_element_generic(
                D::from_usize(n),
                U1,
                T::from_f64(10.0).unwrap(),
            ),
            bounds_cached: false,
            success_history: VecDeque::with_capacity(success_history_size),
            current_mutation_rate: initial_mutation_rate,
            current_crossover_prob: initial_crossover_prob,
            generation_improvements: VecDeque::with_capacity(20),
        }
    }

    // Tune params based on history
    fn adapt_parameters(&mut self, generation_improvement: f64) {
        if !self.conf.common.adaptive_parameters {
            return;
        }

        self.generation_improvements
            .push_back(generation_improvement);
        if self.generation_improvements.len() > 20 {
            self.generation_improvements.pop_front();
        }

        let success_rate = if !self.success_history.is_empty() {
            self.success_history.iter().filter(|&&x| x).count() as f64
                / self.success_history.len() as f64
        } else {
            0.5
        };

        let avg_improvement = if self.generation_improvements.len() > 5 {
            self.generation_improvements.iter().sum::<f64>()
                / self.generation_improvements.len() as f64
        } else {
            generation_improvement
        };

        let adaptation_rate = self.conf.common.adaptation_rate;

        if success_rate < 0.2 || avg_improvement < 1e-6 {
            self.current_mutation_rate *= 1.0 + adaptation_rate; // Low success or stagnation - increase exploration
        } else if success_rate > 0.6 && avg_improvement > 1e-4 {
            self.current_mutation_rate *= 1.0 - adaptation_rate * 0.3; // High success - fine-tune
        }

        self.current_mutation_rate = self.current_mutation_rate.clamp(0.001, 0.5);

        if success_rate < 0.2 {
            self.current_crossover_prob *= 1.0 - adaptation_rate * 0.3; // Low success - reduce crossover slightly
        } else if success_rate > 0.5 {
            self.current_crossover_prob *= 1.0 + adaptation_rate * 0.2; // High success - increase crossover
        }

        self.current_crossover_prob = self.current_crossover_prob.clamp(0.1, 0.95);
    }

    fn track_success(&mut self, old_fitness: T, new_fitness: T) {
        if !self.conf.common.adaptive_parameters {
            return;
        }

        let improved = new_fitness > old_fitness;
        self.success_history.push_back(improved);

        if self.success_history.len() > self.conf.common.success_history_size {
            self.success_history.pop_front();
        }
    }

    pub fn get_current_parameters(&self) -> (f64, f64) {
        (self.current_mutation_rate, self.current_crossover_prob)
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for CGA<T, N, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    N: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<N, D> + Allocator<N> + Allocator<U1, D> + Allocator<D> + Allocator<Dyn>,
{
    fn step(&mut self) {
        let previous_best = self.st.best_f;

        let selected = self
            .selector
            .select(&self.st.pop, &self.st.fitness, &self.st.constraints);
        let mut offspring = self.crossover.crossover(&selected);

        let bounds = if !self.bounds_cached {
            let sample_individual = offspring.row(0).transpose();
            let lower_bounds = self
                .opt_prob
                .objective
                .x_lower_bound(&sample_individual)
                .unwrap_or_else(|| self.cached_bounds_lower.clone());
            let upper_bounds = self
                .opt_prob
                .objective
                .x_upper_bound(&sample_individual)
                .unwrap_or_else(|| self.cached_bounds_upper.clone());

            self.cached_bounds_lower = lower_bounds.clone();
            self.cached_bounds_upper = upper_bounds.clone();
            self.bounds_cached = true;

            (lower_bounds[0], upper_bounds[0])
        } else {
            (self.cached_bounds_lower[0], self.cached_bounds_upper[0])
        };

        let generation = self.st.iter;

        let mutated_rows: Vec<_> = (0..offspring.nrows())
            .into_par_iter()
            .map_init(
                || self.mutation.clone(),
                |mutation, i| {
                    let mut individual =
                        OVector::<T, D>::zeros_generic(D::from_usize(offspring.ncols()), U1);
                    for j in 0..offspring.ncols() {
                        individual[j] = offspring[(i, j)];
                    }

                    mutation.mutate(&individual, bounds, generation)
                },
            )
            .collect();

        for (i, mutated) in mutated_rows.into_iter().enumerate() {
            offspring.set_row(i, &mutated.transpose());
        }

        let (new_fitness, new_constraints): (Vec<T>, Vec<bool>) = (0..offspring.nrows())
            .into_par_iter()
            .map(|i| {
                let individual = offspring.row(i).transpose();
                let fit = self.opt_prob.evaluate(&individual);
                let constr = self.opt_prob.is_feasible(&individual);
                (fit, constr)
            })
            .unzip();

        let mut new_fitness =
            OVector::<T, N>::from_vec_generic(N::from_usize(offspring.nrows()), U1, new_fitness);
        let mut new_constraints = OVector::<bool, N>::from_vec_generic(
            N::from_usize(offspring.nrows()),
            U1,
            new_constraints,
        );

        // Elitism: Keep the best individual from previous generation
        let (best_old_idx, best_old_fitness) = (0..self.st.fitness.len())
            .into_par_iter()
            .filter(|&i| self.st.constraints[i])
            .max_by(|&i, &j| {
                self.st.fitness[i]
                    .partial_cmp(&self.st.fitness[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|idx| (idx, self.st.fitness[idx]))
            .unwrap_or((0, self.st.fitness[0]));

        // Replace worst offspring with best old individual if better
        let (worst_new_idx, worst_new_fitness) = (0..new_fitness.len())
            .into_par_iter()
            .min_by(|&i, &j| {
                new_fitness[i]
                    .partial_cmp(&new_fitness[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|idx| (idx, new_fitness[idx]))
            .unwrap_or((0, new_fitness[0]));

        if best_old_fitness > worst_new_fitness {
            offspring.set_row(worst_new_idx, &self.st.pop.row(best_old_idx));
            new_fitness[worst_new_idx] = best_old_fitness;
            new_constraints[worst_new_idx] = self.st.constraints[best_old_idx];
        }

        self.st.pop = offspring;
        self.st.fitness = new_fitness;
        self.st.constraints = new_constraints;

        let (new_best_idx, new_best_fitness) = (0..self.st.fitness.len())
            .into_par_iter()
            .filter(|&i| self.st.constraints[i])
            .max_by(|&i, &j| {
                self.st.fitness[i]
                    .partial_cmp(&self.st.fitness[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|idx| (idx, self.st.fitness[idx]))
            .unwrap_or((0, self.st.fitness[0]));

        if new_best_fitness > self.st.best_f {
            self.st.best_f = new_best_fitness;
            self.st.best_x = self.st.pop.row(new_best_idx).transpose();
        }

        self.track_success(previous_best, self.st.best_f);
        let generation_improvement = (self.st.best_f - previous_best).to_f64().unwrap_or(0.0);

        self.adapt_parameters(generation_improvement);

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

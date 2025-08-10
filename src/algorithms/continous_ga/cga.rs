use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1};
use rayon::prelude::*;

use crate::utils::config::{CGAConf, CrossoverConf, MutationConf, SelectionConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::continous_ga::{
    crossover::*,
    mutation::{Gaussian, MutationOperator, NonUniform, Polynomial, Uniform},
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
    pub mutation: Box<dyn MutationOperator<T, D> + Send + Sync>,
}

impl<T, N, D> CGA<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
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
    ) -> Self {
        let selector: Box<dyn SelectionOperator<T, N, D> + Send + Sync> = match &conf.selection {
            SelectionConf::RouletteWheel(_) => Box::new(RouletteWheel::new(
                init_pop.nrows(),
                conf.common.num_parents,
            )),
            SelectionConf::Tournament(tournament) => Box::new(Tournament::new(
                init_pop.nrows(),
                conf.common.num_parents,
                tournament.tournament_size,
            )),
            SelectionConf::Residual(_) => {
                Box::new(Residual::new(init_pop.nrows(), conf.common.num_parents))
            }
        };

        let crossover: Box<dyn CrossoverOperator<T, N, D> + Send + Sync> = match &conf.crossover {
            CrossoverConf::Random(random) => {
                Box::new(Random::new(random.crossover_prob, init_pop.nrows()))
            }
            CrossoverConf::Heuristic(heuristic) => {
                Box::new(Heuristic::new(heuristic.crossover_prob, init_pop.nrows()))
            }
        };

        let mutation: Box<dyn MutationOperator<T, D> + Send + Sync> = match &conf.mutation {
            MutationConf::Gaussian(gaussian) => {
                Box::new(Gaussian::new(gaussian.mutation_rate, gaussian.sigma))
            }
            MutationConf::Uniform(uniform) => Box::new(Uniform::new(uniform.mutation_rate)),
            MutationConf::NonUniform(non_uniform) => Box::new(NonUniform::new(
                non_uniform.mutation_rate,
                non_uniform.b,
                max_iter,
            )),
            MutationConf::Polynomial(polynomial) => {
                Box::new(Polynomial::new(polynomial.mutation_rate, polynomial.eta_m))
            }
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
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for CGA<T, N, D>
where
    T: FloatNum,
    D: Dim,
    N: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<N, D> + Allocator<N> + Allocator<U1, D> + Allocator<D> + Allocator<Dyn>,
{
    fn step(&mut self) {
        let selected = self
            .selector
            .select(&self.st.pop, &self.st.fitness, &self.st.constraints);
        let mut offspring = self.crossover.crossover(&selected);

        // Apply mutation
        let fallback_vec_lower = OVector::<T, D>::from_element_generic(
            D::from_usize(offspring.ncols()),
            U1,
            T::from_f64(-10.0).unwrap(),
        );
        let fallback_vec_upper = OVector::<T, D>::from_element_generic(
            D::from_usize(offspring.ncols()),
            U1,
            T::from_f64(10.0).unwrap(),
        );

        let bounds = (
            self.opt_prob
                .objective
                .x_lower_bound(&offspring.row(0).transpose())
                .unwrap_or(fallback_vec_lower)[0],
            self.opt_prob
                .objective
                .x_upper_bound(&offspring.row(0).transpose())
                .unwrap_or(fallback_vec_upper)[0],
        );

        for i in 0..offspring.nrows() {
            let individual = offspring.row(i).transpose();
            let mutated = self.mutation.mutate(&individual, bounds, self.st.iter);
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
        let mut best_old_idx = 0;
        let mut best_old_fitness = self.st.fitness[0];
        for i in 1..self.st.fitness.len() {
            if self.st.fitness[i] > best_old_fitness && self.st.constraints[i] {
                best_old_idx = i;
                best_old_fitness = self.st.fitness[i];
            }
        }

        // Replace worst offspring with best old individual if better
        let mut worst_new_idx = 0;
        let mut worst_new_fitness = new_fitness[0];
        for i in 1..new_fitness.len() {
            if new_fitness[i] < worst_new_fitness {
                worst_new_idx = i;
                worst_new_fitness = new_fitness[i];
            }
        }

        if best_old_fitness > worst_new_fitness {
            offspring.set_row(worst_new_idx, &self.st.pop.row(best_old_idx));
            new_fitness[worst_new_idx] = best_old_fitness;
            new_constraints[worst_new_idx] = self.st.constraints[best_old_idx];
        }

        self.st.pop = offspring;
        self.st.fitness = new_fitness;
        self.st.constraints = new_constraints;

        for i in 0..self.st.fitness.len() {
            if self.st.fitness[i] > self.st.best_f && self.st.constraints[i] {
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

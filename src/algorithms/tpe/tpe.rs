use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rayon::prelude::*;

use crate::utils::alg_conf::tpe_conf::TPEConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::tpe::{acquisition::ExpectedImprovement, kernels::KernelDensityEstimator};

pub struct TPE<T: FloatNum, N: Dim, D: Dim>
where
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: TPEConf,
    pub opt_prob: OptProb<T, D>,
    pub st: State<T, N, D>,

    pub kde_l: KernelDensityEstimator<T, D>,
    pub kde_g: KernelDensityEstimator<T, D>,
    pub acquisition: ExpectedImprovement<D>,

    pub observations: Vec<(OVector<T, D>, T)>,
    pub best_observations: Vec<(OVector<T, D>, T)>,
    pub worst_observations: Vec<(OVector<T, D>, T)>,

    pub iteration: usize,
    pub n_initial_random: usize,
    pub prior_weight: T,

    // Stagnation
    pub stagnation_counter: usize,
    pub last_improvement: T,
    pub last_improvement_iter: usize,
    pub restart_counter: usize,
    pub last_restart_iter: usize,
    pub improvement_history: Vec<T>,
    pub diversity_history: Vec<T>,

    pub bounds_cached: bool,
    pub cached_lower_bounds: OVector<T, D>,
    pub cached_upper_bounds: OVector<T, D>,
    pub stagnation_window: usize,
}

impl<T, N, D> TPE<T, N, D>
where
    T: FloatNum + std::iter::Sum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub fn new(conf: TPEConf, init_pop: OMatrix<T, N, D>, opt_prob: OptProb<T, D>, stagnation_window: usize) -> Self {
        let n = init_pop.ncols();
        let population_size = init_pop.nrows();

        let mut fitness_values = OVector::<T, N>::zeros_generic(N::from_usize(population_size), U1);
        let mut constraint_values =
            OVector::<bool, N>::from_element_generic(N::from_usize(population_size), U1, true);

        for i in 0..population_size {
            let x = init_pop.row(i).transpose();
            fitness_values[i] = opt_prob.evaluate(&x);
            constraint_values[i] = opt_prob.is_feasible(&x);
        }

        let best_idx = fitness_values
            .iter()
            .enumerate()
            .filter(|(i, _)| constraint_values[*i])
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &fitness_values[0]))
            .0;

        let best_x = init_pop.row(best_idx).transpose();
        let best_f = fitness_values[best_idx];

        let st = State {
            best_x: best_x.clone(),
            best_f,
            pop: init_pop.clone(),
            fitness: fitness_values.clone(),
            constraints: constraint_values,
            iter: 1,
        };

        let kde_l = KernelDensityEstimator::new(conf.kernel_type, n);
        let kde_g = KernelDensityEstimator::new(conf.kernel_type, n);
        let acquisition = ExpectedImprovement::<D>::new();

        let mut observations = Vec::new();
        for i in 0..population_size {
            let x = init_pop.row(i).transpose();
            observations.push((x, fitness_values[i]));
        }

        observations.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        let n_best = (observations.len() as f64 * conf.gamma).floor() as usize;
        let best_observations = observations[..n_best].to_vec();
        let worst_observations = observations[n_best..].to_vec();

        Self {
            conf: conf.clone(),
            opt_prob,
            st,
            kde_l,
            kde_g,
            acquisition,
            observations,
            best_observations,
            worst_observations,
            iteration: 1,
            n_initial_random: conf.n_initial_random,
            prior_weight: T::from_f64(conf.prior_weight).unwrap(),
            stagnation_counter: 0,
            last_improvement: T::zero(),
            last_improvement_iter: 0,
            restart_counter: 0,
            last_restart_iter: 1,
            improvement_history: Vec::new(),
            diversity_history: Vec::new(),
            bounds_cached: false,
            cached_lower_bounds: OVector::<T, D>::zeros_generic(D::from_usize(n), U1),
            cached_upper_bounds: OVector::<T, D>::zeros_generic(D::from_usize(n), U1),
            stagnation_window,
        }
    }

    fn sample_candidates(&mut self, n_candidates: usize) -> Vec<OVector<T, D>> {
        if self.iteration <= self.n_initial_random {
            self.sample_random_candidates(n_candidates) // Random sampling phase
        } else {
            self.sample_tpe_candidates(n_candidates) // TPE sampling phase
        }
    }

    fn get_bounds(&mut self, candidate: &OVector<T, D>) -> (OVector<T, D>, OVector<T, D>) {
        if !self.bounds_cached {
            let lower_bounds = self
                .opt_prob
                .objective
                .x_lower_bound(candidate)
                .unwrap_or_else(|| {
                    OVector::<T, D>::from_element_generic(
                        D::from_usize(candidate.len()),
                        U1,
                        T::from_f64(-10.0).unwrap(),
                    )
                });
            let upper_bounds = self
                .opt_prob
                .objective
                .x_upper_bound(candidate)
                .unwrap_or_else(|| {
                    OVector::<T, D>::from_element_generic(
                        D::from_usize(candidate.len()),
                        U1,
                        T::from_f64(10.0).unwrap(),
                    )
                });

            self.cached_lower_bounds = lower_bounds.clone();
            self.cached_upper_bounds = upper_bounds.clone();
            self.bounds_cached = true;

            (lower_bounds, upper_bounds)
        } else {
            (self.cached_lower_bounds.clone(), self.cached_upper_bounds.clone())
        }
    }

    fn sample_random_candidates(&mut self, n_candidates: usize) -> Vec<OVector<T, D>> {
        let n = self.st.pop.ncols();
        
        let sample_candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
        let (lb, ub) = self.get_bounds(&sample_candidate);

        // Uniform sampling over problem domain
        let candidates: Vec<OVector<T, D>> = (0..n_candidates)
            .into_par_iter()
            .map(|_| {
                let mut candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
                for j in 0..n {
                    let range = ub[j] - lb[j];
                    candidate[j] = lb[j] + T::from_f64(rand::random::<f64>()).unwrap() * range;
                }
                candidate
            })
            .collect();

        candidates
    }

    fn sample_tpe_candidates(&mut self, n_candidates: usize) -> Vec<OVector<T, D>> {
        let mut candidates = Vec::with_capacity(n_candidates);
        if !self.best_observations.is_empty() {
            let best_x: Vec<_> = self
                .best_observations
                .iter()
                .map(|(x, _)| x.clone())
                .collect();
            self.kde_l.fit(&best_x);
        }
        if !self.worst_observations.is_empty() {
            let worst_x: Vec<_> = self
                .worst_observations
                .iter()
                .map(|(x, _)| x.clone())
                .collect();
            self.kde_g.fit(&worst_x);
        }

        for _ in 0..n_candidates {
            let candidate = self.sample_candidate_with_acquisition();
            candidates.push(candidate);
        }

        candidates
    }

    fn sample_candidate_with_acquisition(&mut self) -> OVector<T, D> {
        let n = self.st.pop.ncols();
        let num_samples = 100;
        let mut best_candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
        let mut best_ei = T::neg_infinity();

        for _ in 0..num_samples {
            let mut candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
            let (lb, ub) = self.get_bounds(&candidate);
            for j in 0..n {
                let range = ub[j] - lb[j];
                candidate[j] = lb[j] + T::from_f64(rand::random::<f64>()).unwrap() * range;
            }

            let ei = self.acquisition.compute_ei(
                &candidate,
                &self.kde_l,
                &self.kde_g,
                self.prior_weight,
            );

            if ei > best_ei {
                best_ei = ei;
                best_candidate = candidate;
            }
        }

        best_candidate
    }

    fn should_restart(&self) -> bool {
        if !self.conf.advanced.use_restart_strategy {
            return false;
        }

        if self.stagnation_counter >= self.stagnation_window {
            return true;
        }

        if self.iteration - self.last_restart_iter >= self.conf.advanced.restart_frequency {
            return true;
        }

        false
    }

    fn perform_restart(&mut self) {
        let n = self.st.pop.ncols();
        self.kde_l = KernelDensityEstimator::new(self.conf.kernel_type, n);
        self.kde_g = KernelDensityEstimator::new(self.conf.kernel_type, n);

        // Diversify population by adding random perturbations
        let population_size = self.st.pop.nrows();
        for i in 0..population_size {
            let mut x = self.st.pop.row(i).transpose();
            for j in 0..x.len() {
                let perturbation = T::from_f64(rand::random::<f64>() * 2.0 - 1.0).unwrap()
                    * T::from_f64(0.1).unwrap();
                x[j] += perturbation;
            }
            for j in 0..x.len() {
                self.st.pop[(i, j)] = x[j];
            }
            self.st.fitness[i] = self.opt_prob.evaluate(&x);
            self.st.constraints[i] = self.opt_prob.is_feasible(&x);
        }

        self.stagnation_counter = 0;
        self.last_restart_iter = self.iteration;
        self.restart_counter += 1;

        eprintln!(
            "TPE restart triggered after {} iterations without improvement",
            self.stagnation_counter
        );
    }

    // Simple pairwise distance to measure diversity
    fn compute_diversity(&self) -> T {
        let population_size = self.st.pop.nrows();
        if population_size < 2 {
            return T::zero();
        }

        let mut total_distance = T::zero();
        let mut pair_count = 0;

        for i in 0..population_size {
            for j in (i + 1)..population_size {
                let x1 = self.st.pop.row(i).transpose();
                let x2 = self.st.pop.row(j).transpose();
                let distance = self.euclidean_distance(&x1, &x2);
                total_distance = total_distance + distance;
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_distance / T::from_usize(pair_count).unwrap()
        } else {
            T::zero()
        }
    }

    fn euclidean_distance(&self, a: &OVector<T, D>, b: &OVector<T, D>) -> T {
        let mut sum_squared = T::zero();
        for j in 0..a.len() {
            let diff = a[j] - b[j];
            sum_squared = sum_squared + diff * diff;
        }
        sum_squared.sqrt()
    }

    fn update_observations(&mut self, new_observations: Vec<(OVector<T, D>, T)>) {
        self.observations.extend(new_observations);
        self.observations
            .sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

        let n_best = (self.observations.len() as f64 * self.conf.gamma).floor() as usize;
        self.best_observations = self.observations[..n_best].to_vec();
        self.worst_observations = self.observations[n_best..].to_vec();
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for TPE<T, N, D>
where
    T: FloatNum + std::iter::Sum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    fn step(&mut self) {
        if self.should_restart() {
            self.perform_restart();
            return;
        }

        let n_candidates = self.conf.n_ei_candidates;
        let candidates = self.sample_candidates(n_candidates);
        let mut new_observations = Vec::new();
        let mut best_candidate = None;
        let mut best_fitness = T::infinity();

        let candidate_fitnesses: Vec<T> = candidates
            .par_iter()
            .map(|x| self.opt_prob.evaluate(x))
            .collect();

        for (i, fitness) in candidate_fitnesses.iter().enumerate() {
            let x = candidates[i].clone();
            new_observations.push((x.clone(), *fitness));

            if *fitness < best_fitness && self.opt_prob.is_feasible(&x) {
                best_fitness = *fitness;
                best_candidate = Some(x);
            }
        }

        self.update_observations(new_observations);

        if let Some(best_x) = best_candidate {
            if best_fitness < self.st.best_f {
                let improvement = self.st.best_f - best_fitness;
                self.improvement_history.push(improvement);
                self.last_improvement = improvement;
                self.last_improvement_iter = self.iteration;
                self.stagnation_counter = 0;

                self.st.best_x = best_x;
                self.st.best_f = best_fitness;
            } else {
                self.stagnation_counter += 1;
            }
        } else {
            self.stagnation_counter += 1;
        }

        let diversity = self.compute_diversity();
        self.diversity_history.push(diversity);

        let max_history = self.conf.max_history;
        if self.improvement_history.len() > max_history {
            self.improvement_history.remove(0);
        }
        if self.diversity_history.len() > max_history {
            self.diversity_history.remove(0);
        }

        let population_size = self.st.pop.nrows();
        let mut new_pop = OMatrix::<T, N, D>::zeros_generic(
            N::from_usize(population_size),
            D::from_usize(self.st.pop.ncols()),
        );
        let mut new_fitness = OVector::<T, N>::zeros_generic(N::from_usize(population_size), U1);
        let mut new_constraints =
            OVector::<bool, N>::from_element_generic(N::from_usize(population_size), U1, true);

        for (i, (x, fitness)) in self.observations.iter().take(population_size).enumerate() {
            for j in 0..x.len() {
                new_pop[(i, j)] = x[j]; // Fill with best measurements
            }
            new_fitness[i] = *fitness;
            new_constraints[i] = self.opt_prob.is_feasible(x);
        }

        self.st.pop = new_pop;
        self.st.fitness = new_fitness;
        self.st.constraints = new_constraints;
        self.st.iter += 1;
        self.iteration += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

impl<T, N, D> TPE<T, N, D>
where
    T: FloatNum + std::iter::Sum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub fn get_performance_stats(&self) -> (usize, usize, T, T) {
        let avg_improvement = if !self.improvement_history.is_empty() {
            let sum: T = self
                .improvement_history
                .iter()
                .fold(T::zero(), |acc, &x| acc + x);
            sum / T::from_usize(self.improvement_history.len()).unwrap()
        } else {
            T::zero()
        };

        let avg_diversity = if !self.diversity_history.is_empty() {
            let sum: T = self
                .diversity_history
                .iter()
                .fold(T::zero(), |acc, &x| acc + x);
            sum / T::from_usize(self.diversity_history.len()).unwrap()
        } else {
            T::zero()
        };

        (
            self.restart_counter,
            self.stagnation_counter,
            avg_improvement,
            avg_diversity,
        )
    }

    pub fn is_stagnated(&self) -> bool {
        self.stagnation_counter >= self.stagnation_window
    }
}

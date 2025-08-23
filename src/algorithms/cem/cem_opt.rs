use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField, U1};
use num_traits::Float;
use rand::{self, rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::iter::Sum;

use crate::utils::config::CEMConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

pub struct CEM<T, N, D>
where
    T: FloatNum + Send + Sync + nalgebra::ComplexField,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<D, D>,
{
    pub conf: CEMConf,
    pub opt_prob: OptProb<T, D>,
    pub st: State<T, N, D>,

    pub mean: OVector<T, D>,
    pub covariance: OMatrix<T, D, D>,
    pub std_dev: OVector<T, D>,
    pub cached_cholesky: Option<nalgebra::Cholesky<T, D>>,
    pub covariance_changed: bool,

    pub improvement_history: Vec<T>,
    pub diversity_history: Vec<T>,
    pub stagnation_counter: usize,
    pub last_improvement: T,
    pub last_improvement_iter: usize,
    pub restart_counter: usize,
    pub last_restart_iter: usize,
    pub stagnation_window: usize,

    rng: StdRng,
}

impl<T, N, D> CEM<T, N, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D> + Allocator<D, D>,
{
    pub fn new(
        conf: CEMConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        stagnation_window: usize,
        seed: u64,
    ) -> Self {
        let n = init_pop.ncols();
        let population_size = init_pop.nrows();

        let mean = if population_size > 0 {
            let mut mean_vec = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
            for i in 0..n {
                let sum: T = (0..population_size).map(|j| init_pop[(j, i)]).sum();
                mean_vec[i] = sum / T::from_usize(population_size).unwrap();
            }
            mean_vec
        } else {
            OVector::<T, D>::zeros_generic(D::from_usize(n), U1)
        };

        let initial_std = T::from_f64(conf.common.initial_std).unwrap();
        let std_dev = OVector::<T, D>::from_element_generic(D::from_usize(n), U1, initial_std);
        let mut covariance = OMatrix::<T, D, D>::zeros_generic(D::from_usize(n), D::from_usize(n));

        // Init cov as diagonal matrix
        for i in 0..n {
            covariance[(i, i)] = std_dev[i] * std_dev[i];
        }

        let mut st = State {
            best_x: mean.clone(),
            best_f: T::neg_infinity(),
            pop: init_pop.clone(),
            fitness: OVector::<T, N>::zeros_generic(N::from_usize(population_size), U1),
            constraints: OVector::<bool, N>::from_element_generic(
                N::from_usize(population_size),
                U1,
                true,
            ),
            iter: 0,
        };

        let (fitness, constraints): (Vec<T>, Vec<bool>) = (0..population_size)
            .into_par_iter()
            .map(|i| {
                let x = init_pop.row(i).transpose();
                let fit = opt_prob.evaluate(&x);
                let constr = opt_prob.is_feasible(&x);
                (fit, constr)
            })
            .unzip();

        st.fitness = OVector::<T, N>::from_vec_generic(N::from_usize(population_size), U1, fitness);
        st.constraints =
            OVector::<bool, N>::from_vec_generic(N::from_usize(population_size), U1, constraints);

        if let Some((best_idx, _)) = st
            .fitness
            .iter()
            .enumerate()
            .filter(|(i, _)| st.constraints[*i])
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            st.best_x = st.pop.row(best_idx).transpose();
            st.best_f = st.fitness[best_idx];
        }

        Self {
            conf,
            opt_prob,
            st,
            mean,
            covariance,
            std_dev,
            cached_cholesky: None,
            covariance_changed: true,
            improvement_history: Vec::new(),
            diversity_history: Vec::new(),
            stagnation_counter: 0,
            last_improvement: T::neg_infinity(),
            last_improvement_iter: 0,
            restart_counter: 0,
            last_restart_iter: 0,
            stagnation_window,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn get_bounds(&self, candidate: &OVector<T, D>) -> (OVector<T, D>, OVector<T, D>) {
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

        (lower_bounds, upper_bounds)
    }

    fn should_restart(&self) -> bool {
        if !self.conf.advanced.use_restart_strategy {
            return false;
        }

        let restart_freq = self.conf.advanced.restart_frequency;
        let stagnation_threshold = restart_freq / 2;

        // Frequency-based
        let basic_restart = self.stagnation_counter > stagnation_threshold
            || (self.st.iter - self.last_restart_iter) > restart_freq;

        // Based on diversity and convergence
        if self.st.iter > 20 {
            let recent_diversity = self
                .diversity_history
                .iter()
                .rev()
                .take(5)
                .fold(T::zero(), |acc, &x| acc + x)
                / T::from_f64(5.0).unwrap();

            let diversity_threshold = T::from_f64(1e-4).unwrap();
            let diversity_restart = recent_diversity < diversity_threshold;

            return basic_restart || diversity_restart;
        }

        basic_restart
    }

    fn should_early_stop(&self) -> bool {
        if !self.conf.advanced.use_restart_strategy {
            return false;
        }

        if self.stagnation_counter > self.stagnation_window * 2 {
            return true;
        }

        let threshold_window = self.conf.advanced.improvement_threshold_window;
        if self.improvement_history.len() >= threshold_window {
            let recent_improvements =
                &self.improvement_history[self.improvement_history.len() - threshold_window..];
            let avg_improvement: T = recent_improvements.iter().cloned().sum::<T>()
                / T::from_usize(recent_improvements.len()).unwrap();

            if avg_improvement < T::from_f64(1e-8).unwrap() {
                return true;
            }
        }

        false
    }

    fn perform_restart(&mut self) {
        let n = self.mean.len();

        let mean_copy = self.mean.clone();
        let (lb, ub) = self.get_bounds(&mean_copy);
        for i in 0..n {
            let range = ub[i] - lb[i];
            self.mean[i] = lb[i] + T::from_f64(self.rng.random::<f64>()).unwrap() * range;
        }

        let initial_std = T::from_f64(self.conf.common.initial_std).unwrap();
        self.std_dev = OVector::<T, D>::from_element_generic(D::from_usize(n), U1, initial_std);

        // Inject diversity into std
        for i in 0..n {
            let noise = T::from_f64(self.rng.random_range(0.5..1.5)).unwrap();
            self.std_dev[i] *= noise;
        }

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    self.covariance[(i, j)] = self.std_dev[i] * self.std_dev[i];
                } else {
                    self.covariance[(i, j)] = T::zero();
                }
            }
        }
        self.covariance_changed = true;
        self.stagnation_counter = 0;
        self.last_restart_iter = self.st.iter;
        self.restart_counter += 1;

        if self.improvement_history.len() > 10 {
            self.improvement_history
                .drain(0..self.improvement_history.len() - 10);
        }
        if self.diversity_history.len() > 10 {
            self.diversity_history
                .drain(0..self.diversity_history.len() - 10);
        }

        eprintln!(
            "CEM restart triggered after {} iterations without improvement (restart #{})",
            self.st.iter - self.last_improvement_iter,
            self.restart_counter
        );
    }

    fn sample_population(&mut self) -> Vec<OVector<T, D>> {
        let n = self.mean.len();
        let population_size = self.conf.common.population_size;

        let antithetic_ratio = if self.conf.sampling.use_antithetic {
            self.conf.sampling.antithetic_ratio
        } else {
            0.0
        };

        let regular_sample_count = (population_size as f64 / (1.0 + antithetic_ratio)) as usize;
        let antithetic_count = population_size - regular_sample_count;

        let mut population = Vec::with_capacity(population_size);

        for _ in 0..regular_sample_count {
            let sample = self.sample_multivariate_normal();
            population.push(sample);
        }

        for sample in &mut population {
            let (lb, ub) = self.get_bounds(sample);
            for i in 0..n {
                sample[i] = Float::min(Float::max(sample[i], lb[i]), ub[i]);
            }
        }

        // This a variance reduction technique: generated negatively correlated pairs
        if self.conf.sampling.use_antithetic && antithetic_count > 0 {
            for i in 0..antithetic_count.min(regular_sample_count) {
                let antithetic = &self.mean - (&population[i] - &self.mean); // Reflect about mean

                let (lb, ub) = self.get_bounds(&antithetic);
                let mut bounded_antithetic = antithetic;
                for j in 0..n {
                    bounded_antithetic[j] =
                        Float::min(Float::max(bounded_antithetic[j], lb[j]), ub[j]);
                }

                population.push(bounded_antithetic);
            }
        }

        population
    }

    fn sample_multivariate_normal(&mut self) -> OVector<T, D> {
        let n = self.mean.len();

        let mut z = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..n {
            z[i] = T::from_f64(normal.sample(&mut self.rng)).unwrap();
        }

        if self.cached_cholesky.is_none() || self.covariance_changed {
            let mut cov_matrix = self.covariance.clone();
            let reg_factor = T::from_f64(1e-6).unwrap();
            for i in 0..n {
                cov_matrix[(i, i)] += reg_factor;
            }

            let new_cholesky = cov_matrix
                .cholesky()
                .expect("Covariance matrix should be positive definite after regularization");
            self.cached_cholesky = Some(new_cholesky);
            self.covariance_changed = false;
        }

        // Transform standard normal: x = μ + L * z
        let l_matrix = self.cached_cholesky.as_ref().unwrap();
        let transformed = l_matrix.l() * z;
        &self.mean + &transformed
    }

    fn update_distribution(&mut self, elite_samples: &[OVector<T, D>]) {
        if elite_samples.is_empty() {
            return;
        }

        let n = self.mean.len();
        let elite_size = elite_samples.len();

        // Update mean with elite samples
        let mut new_mean = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
        for sample in elite_samples {
            new_mean += sample;
        }
        new_mean /= T::from_usize(elite_size).unwrap();

        // Update covariance with elite samples
        let mut new_covariance =
            OMatrix::<T, D, D>::zeros_generic(D::from_usize(n), D::from_usize(n));

        for sample in elite_samples {
            let diff = sample - &new_mean;
            // Rank-1 update: cov += diff * diff^T
            for i in 0..n {
                for j in 0..n {
                    new_covariance[(i, j)] += diff[i] * diff[j];
                }
            }
        }
        new_covariance /= T::from_usize(elite_size).unwrap();

        // Ensure positive definiteness in cov
        if self.conf.advanced.use_covariance_adaptation {
            let reg = T::from_f64(self.conf.advanced.covariance_regularization).unwrap();
            for i in 0..n {
                new_covariance[(i, i)] += reg;
            }

            // Additional numerical stability: ensure minimum eigenvalue
            let min_diag = new_covariance
                .diagonal()
                .iter()
                .fold(T::infinity(), |acc, &x| Float::min(acc, x));
            if min_diag < reg {
                let additional_reg = reg - min_diag;
                for i in 0..n {
                    new_covariance[(i, i)] += additional_reg;
                }
            }
        }

        // Smooth updates with EMA
        let alpha = T::from_f64(self.conf.adaptation.smoothing_factor).unwrap();

        // Update mean with EMA
        let one_minus_alpha = T::one() - alpha;
        for i in 0..n {
            self.mean[i] = alpha * new_mean[i] + one_minus_alpha * self.mean[i];
        }

        // Update covariance with EMA
        for i in 0..n {
            for j in 0..n {
                self.covariance[(i, j)] =
                    alpha * new_covariance[(i, j)] + one_minus_alpha * self.covariance[(i, j)];
            }
        }
        self.covariance_changed = true;

        // Update std from diagonal of cov
        for i in 0..n {
            let new_std = Float::sqrt(self.covariance[(i, i)]);
            let min_std = T::from_f64(self.conf.common.min_std).unwrap();
            let max_std = T::from_f64(self.conf.common.max_std).unwrap();
            self.std_dev[i] = Float::min(Float::max(new_std, min_std), max_std);
        }
    }

    fn compute_diversity(&self) -> T {
        let n = self.mean.len();
        let population_size = self.st.pop.nrows();

        if population_size == 0 {
            return T::zero();
        }

        // Vectorized diversity computation
        let mut total_variance = T::zero();
        for i in 0..n {
            let mean_val = self.mean[i];
            let variance: T = (0..population_size)
                .map(|j| {
                    let diff = self.st.pop[(j, i)] - mean_val;
                    diff * diff
                })
                .sum();
            total_variance += variance;
        }

        Float::sqrt(total_variance) / T::from_usize(n).unwrap()
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for CEM<T, N, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D> + Allocator<D, D>,
{
    fn step(&mut self) {
        if self.should_early_stop() {
            eprintln!("CEM early stopping triggered due to stagnation");
            return;
        }

        if self.should_restart() {
            self.perform_restart();
            return;
        }

        let population = self.sample_population();
        let population_size = population.len();

        let (fitness, constraints): (Vec<T>, Vec<bool>) = population
            .par_iter()
            .map(|x| {
                let fit = self.opt_prob.evaluate(x);
                let constr = self.opt_prob.is_feasible(x);
                (fit, constr)
            })
            .unzip();

        let sample_dimension = if !population.is_empty() {
            population[0].len()
        } else {
            0
        };

        let mut new_pop = OMatrix::<T, N, D>::zeros_generic(
            N::from_usize(population_size),
            D::from_usize(sample_dimension),
        );

        for (i, sample) in population.iter().enumerate() {
            for (j, &val) in sample.iter().enumerate() {
                new_pop[(i, j)] = val;
            }
        }

        self.st.pop = new_pop;

        self.st.fitness =
            OVector::<T, N>::from_vec_generic(N::from_usize(population_size), U1, fitness.clone());

        self.st.constraints = OVector::<bool, N>::from_vec_generic(
            N::from_usize(population_size),
            U1,
            constraints.clone(),
        );

        let mut best_fitness = T::neg_infinity();
        let mut best_idx = 0;

        for (i, (fit, constr)) in fitness.iter().zip(constraints.iter()).enumerate() {
            if *constr && *fit > best_fitness {
                best_fitness = *fit;
                best_idx = i;
            }
        }

        if best_fitness > self.st.best_f {
            self.st.best_f = best_fitness;
            self.st.best_x = population[best_idx].clone();
            self.last_improvement = best_fitness;
            self.last_improvement_iter = self.st.iter;
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        // Elite samples: top ρ% of feasible solutions
        let mut elite_indices: Vec<usize> =
            (0..population_size).filter(|&i| constraints[i]).collect();

        elite_indices.sort_by(|&a, &b| fitness[b].partial_cmp(&fitness[a]).unwrap());
        let elite_size = self.conf.common.elite_size.min(elite_indices.len());
        let elite_samples: Vec<OVector<T, D>> = elite_indices[..elite_size]
            .iter()
            .map(|&i| population[i].clone())
            .collect();

        self.update_distribution(&elite_samples);
        self.improvement_history.push(best_fitness);
        self.diversity_history.push(self.compute_diversity());

        if best_fitness > self.last_improvement {
            self.last_improvement = best_fitness;
            self.last_improvement_iter = self.st.iter;
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        let max_history = self.conf.advanced.improvement_history_size;
        if self.improvement_history.len() > max_history {
            self.improvement_history
                .drain(0..self.improvement_history.len() - max_history);
            self.diversity_history
                .drain(0..self.diversity_history.len() - max_history);
        }

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

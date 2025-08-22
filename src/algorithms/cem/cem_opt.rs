use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField, U1};
use num_traits::Float;
use rand::{self, Rng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::iter::Sum;

use crate::utils::config::CEMConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

pub struct CEM<T, N, D>
where
    T: FloatNum + Send + Sync,
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

    pub improvement_history: Vec<T>,
    pub diversity_history: Vec<T>,
    pub stagnation_counter: usize,
    pub last_improvement: T,
    pub last_improvement_iter: usize,
    pub restart_counter: usize,
    pub last_restart_iter: usize,
    pub stagnation_window: usize,
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
            improvement_history: Vec::new(),
            diversity_history: Vec::new(),
            stagnation_counter: 0,
            last_improvement: T::neg_infinity(),
            last_improvement_iter: 0,
            restart_counter: 0,
            last_restart_iter: 0,
            stagnation_window,
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

        self.stagnation_counter > stagnation_threshold
            || (self.st.iter - self.last_restart_iter) > restart_freq
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
        let mut rng = rand::rng();

        let mean_copy = self.mean.clone();
        let (lb, ub) = self.get_bounds(&mean_copy);
        for i in 0..n {
            let range = ub[i] - lb[i];
            self.mean[i] = lb[i] + T::from_f64(rng.random::<f64>()).unwrap() * range;
        }

        let initial_std = T::from_f64(self.conf.common.initial_std).unwrap();
        self.std_dev = OVector::<T, D>::from_element_generic(D::from_usize(n), U1, initial_std);

        // Inject diversity into std
        for i in 0..n {
            let noise = T::from_f64(rng.random_range(0.5..1.5)).unwrap();
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

        if self.conf.sampling.use_antithetic && antithetic_count > 0 {
            for i in 0..antithetic_count.min(regular_sample_count) {
                let antithetic = self.mean.clone() - (population[i].clone() - self.mean.clone());

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

    fn sample_multivariate_normal(&self) -> OVector<T, D> {
        let n = self.mean.len();

        // tandard normal samples
        let mut z = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
        for i in 0..n {
            let normal = Normal::new(0.0, 1.0).unwrap();
            z[i] = T::from_f64(normal.sample(&mut rand::rng())).unwrap();
        }

        // Use Cholesky decomposition for multivariate normal: L * L^T = Σ
        let mut cov_matrix = self.covariance.clone();
        let reg_factor = T::from_f64(1e-6).unwrap();
        for i in 0..n {
            cov_matrix[(i, i)] += reg_factor;
        }

        let l_matrix = cov_matrix
            .cholesky()
            .expect("Covariance matrix should be positive definite after regularization");

        // Transform standard normal: x = μ + L * z
        let transformed = l_matrix.l() * z;
        let mut sample = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
        for i in 0..n {
            sample[i] = self.mean[i] + transformed[i];
        }

        sample
    }

    fn update_distribution(&mut self, elite_samples: &[OVector<T, D>]) {
        if elite_samples.is_empty() {
            return;
        }

        let n = self.mean.len();
        let elite_size = elite_samples.len();

        // Update mean with elite samples
        let mut new_mean = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
        for i in 0..n {
            let sum: T = elite_samples.iter().map(|x| x[i]).sum();
            new_mean[i] = sum / T::from_usize(elite_size).unwrap();
        }

        // Update cov with elite samples
        let mut new_covariance =
            OMatrix::<T, D, D>::zeros_generic(D::from_usize(n), D::from_usize(n));
        for i in 0..n {
            for j in 0..n {
                let sum: T = elite_samples
                    .iter()
                    .map(|x| (x[i] - new_mean[i]) * (x[j] - new_mean[j]))
                    .sum();
                new_covariance[(i, j)] = sum / T::from_usize(elite_size).unwrap();
            }
        }

        // Ensure positive definiteness in cov
        if self.conf.advanced.use_covariance_adaptation {
            let reg = T::from_f64(self.conf.advanced.covariance_regularization).unwrap();
            for i in 0..n {
                new_covariance[(i, i)] += reg;
            }
        }

        // Smooth updates with EMA
        let alpha = T::from_f64(self.conf.adaptation.smoothing_factor).unwrap();

        // Update mean with EMA
        for i in 0..n {
            self.mean[i] = alpha * new_mean[i] + (T::one() - alpha) * self.mean[i];
        }

        for i in 0..n {
            for j in 0..n {
                self.covariance[(i, j)] =
                    alpha * new_covariance[(i, j)] + (T::one() - alpha) * self.covariance[(i, j)];
            }
        }

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

        let mut diversity = T::zero();
        for i in 0..n {
            let mean_val = self.mean[i];
            let variance: T = (0..population_size)
                .map(|j| {
                    let diff = self.st.pop[(j, i)] - mean_val;
                    diff * diff
                })
                .sum();
            diversity += variance;
        }

        Float::sqrt(diversity) / T::from_usize(n).unwrap()
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

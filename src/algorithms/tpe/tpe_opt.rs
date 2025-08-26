use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::utils::alg_conf::tpe_conf::{BandwidthMethod, SamplingStrategy, TPEConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::tpe::{
    acquisition::{get_acquisition_function, AcquisitionFunctionPtr},
    kernels::KernelDensityEstimator,
};

pub struct TPE<T: FloatNum, N: Dim, D: Dim>
where
    T: Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: TPEConf,
    pub opt_prob: OptProb<T, D>,
    pub st: State<T, N, D>,
    pub kde_l: KernelDensityEstimator<T, D>, // p(x | D^(l)) - better observations
    pub kde_g: KernelDensityEstimator<T, D>, // p(x | D^(g)) - all observations
    pub acquisition: AcquisitionFunctionPtr<T, D>,
    pub observations: Vec<(OVector<T, D>, T)>,
    pub best_observations: Vec<(OVector<T, D>, T)>,
    pub worst_observations: Vec<(OVector<T, D>, T)>,
    pub iteration: usize,
    pub n_initial_random: usize,
    pub prior_weight: T,
    pub stagnation_counter: usize,
    pub last_improvement: T,
    pub last_improvement_iter: usize,
    pub restart_counter: usize,
    pub last_restart_iter: usize,
    pub improvement_history: Vec<T>,
    pub diversity_history: Vec<T>,
    pub convergence_history: Vec<T>,
    pub current_gamma: T,
    pub gamma_history: Vec<T>,
    pub adaptive_noise_scale: T,
    pub noise_scale_history: Vec<T>,
    pub bounds_cached: bool,
    pub cached_lower_bounds: OVector<T, D>,
    pub cached_upper_bounds: OVector<T, D>,
    pub stagnation_window: usize,
    pub kernel_cache: Vec<T>,
    pub candidate_cache: Vec<OVector<T, D>>,
    pub meta_optimization_history: Vec<(T, T)>, // (gamma, performance)
    pub meta_optimization_iter: usize,
    pub kde_refit_counter: usize,
    pub kde_refit_frequency: usize,
    pub observations_changed: bool,
    pub last_kde_fit_iter: usize,
    pub last_observation_count: usize,
    rng: StdRng,
    seed: u64,
}

impl<T, N, D> TPE<T, N, D>
where
    T: FloatNum + std::iter::Sum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub fn new(
        conf: TPEConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        stagnation_window: usize,
        seed: u64,
    ) -> Self {
        let n = init_pop.ncols();
        let population_size = init_pop.nrows();

        let mut fitness_values = OVector::<T, N>::zeros_generic(N::from_usize(population_size), U1);
        let mut constraint_values =
            OVector::<bool, N>::from_element_generic(N::from_usize(population_size), U1, true);

        let fitness_results: Vec<T> = (0..population_size)
            .into_par_iter()
            .map(|i| {
                let x = init_pop.row(i).transpose();
                opt_prob.evaluate(&x)
            })
            .collect();

        let constraint_results: Vec<bool> = (0..population_size)
            .into_par_iter()
            .map(|i| {
                let x = init_pop.row(i).transpose();
                opt_prob.is_feasible(&x)
            })
            .collect();

        for i in 0..population_size {
            fitness_values[i] = fitness_results[i];
            constraint_values[i] = constraint_results[i];
        }

        let best_idx = fitness_values
            .iter()
            .enumerate()
            .filter(|(i, _)| constraint_values[*i])
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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

        let kde_l = KernelDensityEstimator::new_with_config(
            vec![], // Start with empty
            conf.kernel_type.clone(),
            conf.bandwidth.clone(),
        );
        let kde_g = KernelDensityEstimator::new_with_config(
            vec![], // Start with empty
            conf.kernel_type.clone(),
            conf.bandwidth.clone(),
        );

        let acquisition =
            get_acquisition_function::<T, D>(conf.acquisition.acquisition_type.clone());

        let mut observations = Vec::new();
        for i in 0..population_size {
            let x = init_pop.row(i).transpose();
            observations.push((x, fitness_values[i]));
        }

        observations.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap()); // Descending order for maximization
        let n_best = (observations.len() as f64 * conf.gamma).floor() as usize;
        let best_observations = observations[..n_best].to_vec(); // Top γ-quantile (best)
        let worst_observations = observations[n_best..].to_vec(); // Bottom (1-γ)-quantile (worst)
        let observation_count = observations.len();

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
            convergence_history: Vec::new(),
            current_gamma: T::from_f64(conf.gamma).unwrap(),
            gamma_history: Vec::new(),
            adaptive_noise_scale: T::from_f64(conf.sampling.noise_scale).unwrap(),
            noise_scale_history: Vec::new(),
            bounds_cached: false,
            cached_lower_bounds: OVector::<T, D>::zeros_generic(D::from_usize(n), U1),
            cached_upper_bounds: OVector::<T, D>::zeros_generic(D::from_usize(n), U1),
            stagnation_window,
            kernel_cache: Vec::new(),
            candidate_cache: Vec::new(),
            meta_optimization_history: Vec::new(),
            meta_optimization_iter: 0,
            kde_refit_counter: 0,
            kde_refit_frequency: conf.kde_refit_frequency,
            observations_changed: false,
            last_kde_fit_iter: 0,
            last_observation_count: observation_count,
            rng: StdRng::seed_from_u64(seed),
            seed,
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
            (
                self.cached_lower_bounds.clone(),
                self.cached_upper_bounds.clone(),
            )
        }
    }

    fn sample_random_candidates(&mut self, n_candidates: usize) -> Vec<OVector<T, D>> {
        let n = self.st.pop.ncols();
        let (lb, ub) = self.get_bounds(&OVector::<T, D>::zeros_generic(D::from_usize(n), U1));

        let candidates: Vec<OVector<T, D>> = (0..n_candidates)
            .into_par_iter()
            .map_init(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    StdRng::seed_from_u64(
                        self.seed + self.iteration as u64 * 1000 + thread_id as u64,
                    )
                },
                |rng, _| {
                    let mut candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);

                    for j in 0..n {
                        let range = ub[j] - lb[j];
                        candidate[j] = lb[j] + T::from_f64(rng.random::<f64>()).unwrap() * range;
                    }
                    candidate
                },
            )
            .collect();

        candidates
    }

    fn sample_tpe_candidates(&mut self, n_candidates: usize) -> Vec<OVector<T, D>> {
        let mut candidates = Vec::with_capacity(n_candidates);

        if self.should_refit_kdes() {
            self.fit_kdes();
        }

        match self.conf.sampling.strategy {
            SamplingStrategy::Random => {
                candidates = self.sample_random_candidates(n_candidates);
            }
            SamplingStrategy::KDEBased => {
                for _ in 0..n_candidates {
                    let candidate = self.sample_candidate_with_acquisition();
                    candidates.push(candidate);
                }
            }
            SamplingStrategy::Thompson => {
                candidates = self.sample_thompson_candidates(n_candidates);
            }
            SamplingStrategy::Hybrid => {
                candidates = self.sample_hybrid_candidates(n_candidates);
            }
        }

        candidates
    }

    // Refit when performance degraded or observations changed significantly
    fn should_refit_kdes(&self) -> bool {
        self.last_kde_fit_iter == 0
            || self.observations_changed
            || (self.iteration - self.last_kde_fit_iter) >= self.kde_refit_frequency
            || self.observations.len() != self.last_observation_count
    }

    fn fit_kdes(&mut self) {
        // Fit KDE for better observations (l(x))
        if !self.best_observations.is_empty() {
            let best_x: Vec<_> = self
                .best_observations
                .iter()
                .map(|(x, _)| x.clone())
                .collect();
            self.kde_l.fit(&best_x);
        }

        // Fit KDE for all observations (g(x))
        if !self.observations.is_empty() {
            let all_x: Vec<_> = self.observations.iter().map(|(x, _)| x.clone()).collect();
            self.kde_g.fit(&all_x);
        }

        self.last_kde_fit_iter = self.iteration;
        self.last_observation_count = self.observations.len();
        self.observations_changed = false;
    }

    // Sample posterior with Thompson; fallback random
    fn sample_thompson_candidates(&mut self, n_candidates: usize) -> Vec<OVector<T, D>> {
        let n = self.st.pop.ncols();
        let mut candidates = Vec::with_capacity(n_candidates);

        for _ in 0..n_candidates {
            let mut candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);

            if !self.best_observations.is_empty() {
                let random_idx =
                    (self.rng.random::<f64>() * (self.best_observations.len() as f64)) as usize;
                let base_point = &self.best_observations[random_idx].0;

                for j in 0..n {
                    let noise = T::from_f64((self.rng.random::<f64>() * 2.0) - 1.0).unwrap()
                        * self.adaptive_noise_scale;
                    candidate[j] = base_point[j] + noise;
                }

                let (lb, ub) = self.get_bounds(&candidate);
                for j in 0..n {
                    candidate[j] = candidate[j].max(lb[j]).min(ub[j]);
                }
            } else {
                let (lb, ub) = self.get_bounds(&candidate);
                for j in 0..n {
                    let range = ub[j] - lb[j];
                    candidate[j] = lb[j] + T::from_f64(self.rng.random::<f64>()).unwrap() * range;
                }
            }

            candidates.push(candidate);
        }

        candidates
    }

    fn sample_hybrid_candidates(&mut self, n_candidates: usize) -> Vec<OVector<T, D>> {
        let mut candidates = Vec::with_capacity(n_candidates);

        let n_kde = n_candidates / 2;
        let n_thompson = n_candidates - n_kde;

        // KDE-based
        for _ in 0..n_kde {
            let candidate = self.sample_candidate_with_acquisition();
            candidates.push(candidate);
        }

        // Thompson
        let thompson_candidates = self.sample_thompson_candidates(n_thompson);
        candidates.extend(thompson_candidates);

        candidates
    }

    fn sample_candidate_with_acquisition(&mut self) -> OVector<T, D> {
        let n = self.st.pop.ncols();
        let num_samples = 100;
        let mut best_candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
        let mut best_acquisition = T::neg_infinity();

        // Sample candidate from g(x) distribution, fallback uniform
        if !self.best_observations.is_empty() {
            for _ in 0..num_samples {
                let candidate = self.sample_from_good_distribution();

                let acquisition_value = (self.acquisition)(
                    &candidate,
                    &self.kde_l,
                    &self.kde_g,
                    T::from_f64(self.conf.acquisition.kappa).unwrap(),
                );

                if acquisition_value > best_acquisition {
                    best_acquisition = acquisition_value;
                    best_candidate = candidate;
                }
            }
        } else {
            let (lb, ub) = self.get_bounds(&OVector::<T, D>::zeros_generic(D::from_usize(n), U1));
            for j in 0..n {
                let range = ub[j] - lb[j];
                best_candidate[j] = lb[j] + T::from_f64(self.rng.random::<f64>()).unwrap() * range;
            }
        }

        if self.conf.sampling.local_search {
            best_candidate = self.apply_local_search(&best_candidate);
        }

        best_candidate
    }

    fn apply_local_search(&mut self, candidate: &OVector<T, D>) -> OVector<T, D> {
        let mut best_candidate = candidate.clone();
        let mut best_fitness = self.opt_prob.evaluate(candidate);
        let n = candidate.len();
        let (lb, ub) = self.get_bounds(candidate);

        for _ in 0..self.conf.sampling.local_search_steps {
            let mut new_candidate = best_candidate.clone();

            for j in 0..n {
                let perturbation = T::from_f64((self.rng.random::<f64>() * 2.0) - 1.0).unwrap()
                    * self.adaptive_noise_scale
                    * T::from_f64(0.1).unwrap();
                new_candidate[j] += perturbation;

                new_candidate[j] = new_candidate[j].max(lb[j]).min(ub[j]);
            }

            if self.opt_prob.is_feasible(&new_candidate) {
                let new_fitness = self.opt_prob.evaluate(&new_candidate);
                if new_fitness > best_fitness {
                    best_candidate = new_candidate;
                    best_fitness = new_fitness;
                }
            }
        }

        best_candidate
    }

    fn sample_from_good_distribution(&mut self) -> OVector<T, D> {
        let n = self.st.pop.ncols();
        let mut candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);

        // KDE approx with adaptive noise around initial best
        if !self.best_observations.is_empty() {
            let random_idx =
                (self.rng.random::<f64>() * (self.best_observations.len() as f64)) as usize;
            let base_point = &self.best_observations[random_idx].0;

            for j in 0..n {
                let noise = T::from_f64((self.rng.random::<f64>() * 2.0) - 1.0).unwrap()
                    * self.adaptive_noise_scale;
                candidate[j] = base_point[j] + noise;
            }

            let (lb, ub) = self.get_bounds(&candidate);
            for j in 0..n {
                candidate[j] = candidate[j].max(lb[j]).min(ub[j]);
            }
        }

        candidate
    }

    fn sample_random_feasible_point(&mut self) -> OVector<T, D> {
        let n = self.st.pop.ncols();
        let max_attempts = 10;

        for _ in 0..max_attempts {
            let mut candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
            let (lb, ub) = self.get_bounds(&candidate);

            for j in 0..n {
                let range = ub[j] - lb[j];
                candidate[j] = lb[j] + T::from_f64(self.rng.random::<f64>()).unwrap() * range;
            }

            if self.opt_prob.is_feasible(&candidate) {
                return candidate;
            }
        }

        // Fallback: return random in-domain point
        let mut candidate = OVector::<T, D>::zeros_generic(D::from_usize(n), U1);
        let (lb, ub) = self.get_bounds(&candidate);
        for j in 0..n {
            let range = ub[j] - lb[j];
            candidate[j] = lb[j] + T::from_f64(self.rng.random::<f64>()).unwrap() * range;
        }
        candidate
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
        let _n = self.st.pop.ncols();

        self.kde_l = KernelDensityEstimator::new_with_config(
            vec![], // Reinit with empty observations
            self.conf.kernel_type.clone(),
            self.conf.bandwidth.clone(),
        );
        self.kde_g = KernelDensityEstimator::new_with_config(
            vec![], // Reinit with empty observations
            self.conf.kernel_type.clone(),
            self.conf.bandwidth.clone(),
        );

        // Diversify with random perturbations
        let population_size = self.st.pop.nrows();
        for i in 0..population_size {
            let mut x = self.st.pop.row(i).transpose();
            for j in 0..x.len() {
                let perturbation = T::from_f64((self.rng.random::<f64>() * 2.0) - 1.0).unwrap()
                    * self.adaptive_noise_scale;
                x[j] += perturbation;
            }

            let (lb, ub) = self.get_bounds(&x);
            for j in 0..x.len() {
                x[j] = x[j].max(lb[j]).min(ub[j]);
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

    fn should_early_stop(&self) -> bool {
        if !self.conf.advanced.use_early_stopping {
            return false;
        }

        if self.improvement_history.len() < self.conf.advanced.early_stopping_patience {
            return false;
        }

        let recent_improvements = &self.improvement_history
            [self.improvement_history.len() - self.conf.advanced.early_stopping_patience..];

        let avg_improvement: T = recent_improvements.iter().cloned().sum::<T>()
            / T::from_usize(recent_improvements.len()).unwrap();

        avg_improvement < T::from_f64(1e-6).unwrap()
    }

    fn update_adaptive_parameters(&mut self) {
        if self.conf.advanced.use_adaptive_gamma {
            self.update_adaptive_gamma();
        }

        if self.conf.sampling.adaptive_noise {
            self.update_adaptive_noise_scale();
        }

        if matches!(self.conf.bandwidth.method, BandwidthMethod::Adaptive) {
            self.kde_l.update_bandwidths();
            self.kde_g.update_bandwidths();
        }
    }

    fn update_adaptive_gamma(&mut self) {
        let recent_improvements: Vec<T> = self
            .improvement_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        if !recent_improvements.is_empty() {
            let avg_improvement: T = recent_improvements.iter().cloned().sum::<T>()
                / T::from_usize(recent_improvements.len()).unwrap();

            if avg_improvement > T::from_f64(1e-3).unwrap() {
                self.current_gamma = (self.current_gamma * T::from_f64(0.95).unwrap())
                    .max(T::from_f64(0.1).unwrap()); // increase exploitation (decrease gamma)
            } else {
                self.current_gamma = (self.current_gamma * T::from_f64(1.05).unwrap())
                    .min(T::from_f64(0.5).unwrap()); // increase exploration (increase gamma)
            }
        }

        self.gamma_history.push(self.current_gamma);
    }

    fn update_adaptive_noise_scale(&mut self) {
        let recent_diversity: Vec<T> = self
            .diversity_history
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        if !recent_diversity.is_empty() {
            let avg_diversity: T = recent_diversity.iter().cloned().sum::<T>()
                / T::from_usize(recent_diversity.len()).unwrap();

            if avg_diversity < T::from_f64(0.1).unwrap() {
                self.adaptive_noise_scale *= T::from_f64(1.1).unwrap(); // increase exploration
            } else if avg_diversity > T::from_f64(1.0).unwrap() {
                self.adaptive_noise_scale *= T::from_f64(0.9).unwrap(); // decrease exploration
            }

            self.adaptive_noise_scale = self
                .adaptive_noise_scale
                .max(T::from_f64(0.01).unwrap())
                .min(T::from_f64(1.0).unwrap());
        }

        self.noise_scale_history.push(self.adaptive_noise_scale);
    }

    fn perform_meta_optimization(&mut self) {
        if !self.conf.advanced.use_meta_optimization {
            return;
        }

        // Meta-optimize gamma
        if self.iteration % self.conf.advanced.meta_optimization_frequency == 0 {
            let current_performance = self.compute_performance_metric();
            self.meta_optimization_history
                .push((self.current_gamma, current_performance));

            // Simple hill climbing on gamma
            let gamma_candidates = [
                self.current_gamma * T::from_f64(0.9).unwrap(),
                self.current_gamma * T::from_f64(1.1).unwrap(),
                self.current_gamma,
            ];

            let mut best_gamma = self.current_gamma;
            let mut best_performance = current_performance;

            for &gamma in &gamma_candidates {
                if gamma >= T::from_f64(0.1).unwrap() && gamma <= T::from_f64(0.5).unwrap() {
                    let old_gamma = self.current_gamma;
                    self.current_gamma = gamma;

                    let performance = self.compute_performance_metric();
                    if performance > best_performance {
                        best_performance = performance;
                        best_gamma = gamma;
                    }

                    self.current_gamma = old_gamma;
                }
            }

            self.current_gamma = best_gamma;
            self.meta_optimization_iter += 1;
        }
    }

    fn compute_performance_metric(&self) -> T {
        let recent_improvements: Vec<T> = self
            .improvement_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        let recent_diversity: Vec<T> = self
            .diversity_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        let avg_improvement: T = if !recent_improvements.is_empty() {
            recent_improvements.iter().cloned().sum::<T>()
                / T::from_usize(recent_improvements.len()).unwrap()
        } else {
            T::zero()
        };

        let avg_diversity: T = if !recent_diversity.is_empty() {
            recent_diversity.iter().cloned().sum::<T>()
                / T::from_usize(recent_diversity.len()).unwrap()
        } else {
            T::zero()
        };

        // Weighted combination
        avg_improvement * T::from_f64(0.7).unwrap() + avg_diversity * T::from_f64(0.3).unwrap()
    }

    fn compute_diversity(&self) -> T {
        let population_size = self.st.pop.nrows();
        if population_size < 2 {
            return T::zero();
        }

        let total_distance: T = (0..population_size)
            .into_par_iter()
            .map(|i| {
                let mut row_distance = T::zero();
                let x1 = self.st.pop.row(i).transpose();

                for j in (i + 1)..population_size {
                    let x2 = self.st.pop.row(j).transpose();
                    let distance = self.euclidean_distance(&x1, &x2);
                    row_distance += distance;
                }

                row_distance
            })
            .sum();

        let pair_count = T::from_usize(population_size * (population_size - 1) / 2).unwrap();
        total_distance / pair_count
    }

    fn euclidean_distance(&self, a: &OVector<T, D>, b: &OVector<T, D>) -> T {
        let mut sum_squared = T::zero();
        for j in 0..a.len() {
            let diff = a[j] - b[j];
            sum_squared += diff * diff;
        }
        sum_squared.sqrt()
    }

    fn update_observations(&mut self, new_observations: Vec<(OVector<T, D>, T)>) {
        self.observations.extend(new_observations);
        self.observations
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        // Use adaptive gamma for quantile split
        let n_best = (self.observations.len() as f64 * self.current_gamma.to_f64().unwrap()).floor()
            as usize;
        self.best_observations = self.observations[..n_best].to_vec();
        self.worst_observations = self.observations[n_best..].to_vec();
        self.observations_changed = true;
    }

    fn update_convergence_metrics(&mut self) {
        let recent_improvements: Vec<T> = self
            .improvement_history
            .iter()
            .rev()
            .take(20)
            .cloned()
            .collect();

        if !recent_improvements.is_empty() {
            let avg_improvement: T = recent_improvements.iter().cloned().sum::<T>()
                / T::from_usize(recent_improvements.len()).unwrap();

            let convergence_metric = if avg_improvement > T::from_f64(1e-3).unwrap() {
                T::one() // Good progress
            } else if avg_improvement > T::from_f64(1e-6).unwrap() {
                T::from_f64(0.5).unwrap() // Moderate progress
            } else {
                T::zero() // Stagnated
            };

            self.convergence_history.push(convergence_metric);
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for TPE<T, N, D>
where
    T: FloatNum + std::iter::Sum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D>,
{
    fn step(&mut self) {
        if self.should_early_stop() {
            eprintln!("TPE early stopping triggered");
            return;
        }

        if self.should_restart() {
            self.perform_restart();
            return;
        }

        self.perform_meta_optimization();

        let n_candidates = self.conf.n_ei_candidates;
        let candidates = self.sample_candidates(n_candidates);
        let mut new_observations = Vec::new();
        let mut best_candidate = None;
        let mut best_fitness = T::neg_infinity();

        let candidate_fitnesses: Vec<T> = candidates
            .par_iter()
            .map(|x| self.opt_prob.evaluate(x))
            .collect();

        let candidate_feasibility: Vec<bool> = candidates
            .par_iter()
            .map(|x| self.opt_prob.is_feasible(x))
            .collect();

        for (i, (fitness, is_feasible)) in candidate_fitnesses
            .iter()
            .zip(candidate_feasibility.iter())
            .enumerate()
        {
            let x = candidates[i].clone();
            if *is_feasible {
                new_observations.push((x.clone(), *fitness));
            }
            if *fitness > best_fitness && *is_feasible {
                best_fitness = *fitness;
                best_candidate = Some(x);
            }
        }

        self.update_observations(new_observations);

        if let Some(best_x) = best_candidate {
            if best_fitness > self.st.best_f {
                let improvement = best_fitness - self.st.best_f;
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
        self.update_convergence_metrics();

        self.update_adaptive_parameters();

        let max_history = self.conf.max_history;
        if self.improvement_history.len() > max_history {
            self.improvement_history.remove(0);
        }
        if self.diversity_history.len() > max_history {
            self.diversity_history.remove(0);
        }
        if self.convergence_history.len() > max_history {
            self.convergence_history.remove(0);
        }
        if self.gamma_history.len() > max_history {
            self.gamma_history.remove(0);
        }
        if self.noise_scale_history.len() > max_history {
            self.noise_scale_history.remove(0);
        }

        let population_size = self.st.pop.nrows();
        let mut new_pop = OMatrix::<T, N, D>::zeros_generic(
            N::from_usize(population_size),
            D::from_usize(self.st.pop.ncols()),
        );
        let mut new_fitness = OVector::<T, N>::zeros_generic(N::from_usize(population_size), U1);
        let mut new_constraints =
            OVector::<bool, N>::from_element_generic(N::from_usize(population_size), U1, true);

        let mut feasible_count = 0;
        for (x, fitness) in self.observations.iter() {
            if feasible_count >= population_size {
                break;
            }

            if self.opt_prob.is_feasible(x) {
                for j in 0..x.len() {
                    new_pop[(feasible_count, j)] = x[j];
                }
                new_fitness[feasible_count] = *fitness;
                new_constraints[feasible_count] = true;
                feasible_count += 1;
            }
        }

        // Fill remaining slots with random feasible points
        while feasible_count < population_size {
            let random_point = self.sample_random_feasible_point();
            for j in 0..random_point.len() {
                new_pop[(feasible_count, j)] = random_point[j];
            }
            new_fitness[feasible_count] = self.opt_prob.evaluate(&random_point);
            new_constraints[feasible_count] = true;
            feasible_count += 1;
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
    T: FloatNum + std::iter::Sum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub fn get_performance_stats(&self) -> (usize, usize, T, T, T, T, T) {
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

        let avg_convergence = if !self.convergence_history.is_empty() {
            let sum: T = self
                .convergence_history
                .iter()
                .fold(T::zero(), |acc, &x| acc + x);
            sum / T::from_usize(self.convergence_history.len()).unwrap()
        } else {
            T::zero()
        };

        let avg_gamma = if !self.gamma_history.is_empty() {
            let sum: T = self.gamma_history.iter().fold(T::zero(), |acc, &x| acc + x);
            sum / T::from_usize(self.gamma_history.len()).unwrap()
        } else {
            self.current_gamma
        };

        let avg_noise_scale = if !self.noise_scale_history.is_empty() {
            let sum: T = self
                .noise_scale_history
                .iter()
                .fold(T::zero(), |acc, &x| acc + x);
            sum / T::from_usize(self.noise_scale_history.len()).unwrap()
        } else {
            self.adaptive_noise_scale
        };

        (
            self.restart_counter,
            self.stagnation_counter,
            avg_improvement,
            avg_diversity,
            avg_convergence,
            avg_gamma,
            avg_noise_scale,
        )
    }

    pub fn is_stagnated(&self) -> bool {
        self.stagnation_counter >= self.stagnation_window
    }

    pub fn get_current_gamma(&self) -> T {
        self.current_gamma
    }

    pub fn get_adaptive_noise_scale(&self) -> T {
        self.adaptive_noise_scale
    }

    pub fn get_meta_optimization_stats(&self) -> (usize, Vec<(T, T)>) {
        (
            self.meta_optimization_iter,
            self.meta_optimization_history.clone(),
        )
    }
}

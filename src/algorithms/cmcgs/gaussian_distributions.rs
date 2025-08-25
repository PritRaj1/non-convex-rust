use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rand::prelude::*;

use crate::utils::opt_prob::FloatNumber as FloatNum;

/// Gaussian state distribution p_q(s)
#[derive(Clone, Debug)]
pub struct GaussianStateDistribution<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub mean: OVector<T, D>,
    pub variance: OVector<T, D>, // Diagonal cov
}

impl<T, D> GaussianStateDistribution<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    pub fn new(dimension: usize) -> Self {
        let mean = OVector::zeros_generic(D::from_usize(dimension), nalgebra::U1);
        let variance = OVector::from_element_generic(
            D::from_usize(dimension),
            nalgebra::U1,
            T::from_f64(1.0).unwrap(),
        );

        Self { mean, variance }
    }

    pub fn from_states(states: &[OVector<T, D>]) -> Self {
        if states.is_empty() {
            return Self::new(states[0].len());
        }

        let dimension = states[0].len();
        let mut mean = OVector::zeros_generic(D::from_usize(dimension), nalgebra::U1);
        let mut variance = OVector::zeros_generic(D::from_usize(dimension), nalgebra::U1);

        for state in states {
            mean += state;
        }
        mean /= T::from_usize(states.len()).unwrap();

        for state in states {
            let diff = state - &mean;
            for i in 0..dimension {
                variance[i] += diff[i] * diff[i];
            }
        }
        variance /= T::from_usize(states.len()).unwrap();

        for i in 0..dimension {
            if variance[i] < T::from_f64(1e-6).unwrap() {
                variance[i] = T::from_f64(1e-6).unwrap();
            }
        }

        Self { mean, variance }
    }

    // State distribution p_q(s)
    pub fn probability_density(&self, state: &OVector<T, D>) -> T {
        let diff = state - &self.mean;
        let mut log_prob = T::zero();

        for i in 0..state.len() {
            let std_dev = self.variance[i].sqrt();
            let normalized_diff = diff[i] / std_dev;
            log_prob -= T::from_f64(0.5).unwrap() * normalized_diff * normalized_diff;
            log_prob -= std_dev.ln();
        }

        log_prob -= T::from_f64(0.5).unwrap()
            * T::from_f64(2.0 * std::f64::consts::PI).unwrap().ln()
            * T::from_usize(state.len()).unwrap();

        log_prob.exp()
    }

    pub fn update_from_states(&mut self, states: &[OVector<T, D>]) {
        if states.is_empty() {
            return;
        }

        let new_dist = Self::from_states(states);

        // Exponential moving average update
        let alpha = T::from_f64(0.1).unwrap();
        for i in 0..self.mean.len() {
            self.mean[i] = alpha * new_dist.mean[i] + (T::one() - alpha) * self.mean[i];
            self.variance[i] = alpha * new_dist.variance[i] + (T::one() - alpha) * self.variance[i];
        }
    }

    pub fn distance_to_centroid(&self, state: &OVector<T, D>) -> T {
        let diff = state - &self.mean;
        let mut distance_squared = T::zero();

        for i in 0..state.len() {
            distance_squared += diff[i] * diff[i];
        }

        distance_squared.sqrt()
    }

    pub fn sample(&self, rng: &mut impl Rng) -> OVector<T, D> {
        let mut sample = OVector::zeros_generic(D::from_usize(self.mean.len()), nalgebra::U1);

        for i in 0..self.mean.len() {
            let std_dev = self.variance[i].sqrt();
            let noise = T::from_f64(rng.random_range(-2.0..2.0)).unwrap() * std_dev;
            sample[i] = self.mean[i] + noise;
        }

        sample
    }
}

/// Gaussian action policy π_q(a)
#[derive(Clone, Debug)]
pub struct GaussianActionPolicy<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub mean: OVector<T, D>,
    pub variance: OVector<T, D>, // Diagonal cov
    pub prior_alpha: T,          // α_prior
    pub prior_beta: T,           // β_prior
}

impl<T, D> GaussianActionPolicy<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    // Init N(μ = 1/2(a_min + a_max), σ = 1/2(a_max - a_min))
    pub fn new(dimension: usize, action_bounds: (T, T)) -> Self {
        let (a_min, a_max) = action_bounds;
        let mean_val = (a_min + a_max) / T::from_f64(2.0).unwrap();
        let std_val = (a_max - a_min) / T::from_f64(2.0).unwrap();

        let mean = OVector::from_element_generic(D::from_usize(dimension), nalgebra::U1, mean_val);
        let variance = OVector::from_element_generic(
            D::from_usize(dimension),
            nalgebra::U1,
            std_val * std_val,
        );

        Self {
            mean,
            variance,
            prior_alpha: T::from_f64(2.0).unwrap(), // α_prior
            prior_beta: T::from_f64(1.0).unwrap(),  // β_prior
        }
    }

    pub fn update_from_elite_experiences(&mut self, actions: &[OVector<T, D>], returns: &[T]) {
        if actions.is_empty() {
            return;
        }

        let mut indexed_returns: Vec<_> = returns.iter().enumerate().collect();
        indexed_returns.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        let elite_count = (actions.len() / 2).max(1); // Top 50% experiences
        let elite_indices: Vec<_> = indexed_returns
            .iter()
            .take(elite_count)
            .map(|(i, _)| *i)
            .collect();

        let mut new_mean = OVector::zeros_generic(D::from_usize(self.mean.len()), nalgebra::U1);
        for &idx in &elite_indices {
            new_mean += &actions[idx];
        }
        new_mean /= T::from_usize(elite_indices.len()).unwrap();

        // Update variance by Bayesian rule with inverse gamma prior
        let mut new_variance = OVector::zeros_generic(D::from_usize(self.mean.len()), nalgebra::U1);

        for i in 0..self.mean.len() {
            let mut sum_squared_diff = T::zero();
            for &idx in &elite_indices {
                let diff = actions[idx][i] - new_mean[i];
                sum_squared_diff += diff * diff;
            }

            // Posterior hyperparameters: α_posterior = α_prior + n/2, β_posterior = β_prior + sum/2
            let n = T::from_usize(elite_indices.len()).unwrap();
            let alpha_posterior = self.prior_alpha + n / T::from_f64(2.0).unwrap();
            let beta_posterior = self.prior_beta + sum_squared_diff / T::from_f64(2.0).unwrap();

            // New variance = mean of inverse gamma posterior: β_posterior / (α_posterior - 1)
            new_variance[i] = beta_posterior / (alpha_posterior - T::one());
        }

        // Exponential moving average update
        let alpha = T::from_f64(0.1).unwrap();
        for i in 0..self.mean.len() {
            self.mean[i] = alpha * new_mean[i] + (T::one() - alpha) * self.mean[i];
            self.variance[i] = alpha * new_variance[i] + (T::one() - alpha) * self.variance[i];
        }

        // Ensure minimum to prevent collapse
        for i in 0..self.variance.len() {
            if self.variance[i] < T::from_f64(1e-6).unwrap() {
                self.variance[i] = T::from_f64(1e-6).unwrap();
            }
        }
    }

    pub fn sample_action(&self, rng: &mut impl Rng) -> OVector<T, D> {
        let mut action = OVector::zeros_generic(D::from_usize(self.mean.len()), nalgebra::U1);

        for i in 0..self.mean.len() {
            let std_dev = self.variance[i].sqrt();
            let noise = T::from_f64(rng.random_range(-2.0..2.0)).unwrap() * std_dev;
            action[i] = self.mean[i] + noise;
        }

        action
    }

    pub fn get_mean(&self) -> &OVector<T, D> {
        &self.mean
    }

    pub fn get_variance(&self) -> &OVector<T, D> {
        &self.variance
    }
}

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DimSub, OMatrix, OVector, RealField};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::algorithms::parallel_tempering::metropolis_hastings::MetropolisHastings;
use crate::utils::opt_prob::FloatNumber as FloatNum;

/// Manages replica exchange (swapping) operations in parallel tempering
pub struct SwapManager<T, N, D>
where
    T: FloatNum + RealField + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync + DimSub<nalgebra::Const<1>>,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N, D>
        + Allocator<N>
        + Allocator<D, D>
        + Allocator<<D as DimSub<nalgebra::Const<1>>>::Output>,
{
    metropolis_hastings: MetropolisHastings<T, D>,
    swap_acceptance_rates: Vec<f64>,
    adaptive_swapping: bool,
    random_swap_probability: f64,
    swap_rate_smoothing: f64,
    _phantom: std::marker::PhantomData<N>,
    rng: StdRng,
}

impl<T, N, D> SwapManager<T, N, D>
where
    T: FloatNum + RealField + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync + DimSub<nalgebra::Const<1>>,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N, D>
        + Allocator<N>
        + Allocator<D, D>
        + Allocator<<D as DimSub<nalgebra::Const<1>>>::Output>,
{
    pub fn new(
        metropolis_hastings: MetropolisHastings<T, D>,
        num_replicas: usize,
        adaptive_swapping: bool,
        random_swap_probability: f64,
        swap_rate_smoothing: f64,
        seed: u64,
    ) -> Self {
        Self {
            metropolis_hastings,
            swap_acceptance_rates: vec![0.3; num_replicas.saturating_sub(1)],
            adaptive_swapping,
            random_swap_probability,
            swap_rate_smoothing,
            _phantom: std::marker::PhantomData,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn swap_adjacent_replicas(
        &mut self,
        populations: &mut [OMatrix<T, N, D>],
        fitnesses: &mut [OVector<T, N>],
        constraints: &mut [OVector<bool, N>],
        step_sizes: &mut [Vec<OMatrix<T, D, D>>],
        temperatures: &[T],
    ) {
        let n = populations.len();
        if n < 2 {
            return;
        }

        for i in 0..n - 1 {
            let t_i = temperatures[i];
            let t_j = temperatures[i + 1];

            let swap_accepted = self.metropolis_hastings.accept_replica_exchange::<N>(
                &fitnesses[i],
                &fitnesses[i + 1],
                t_i,
                t_j,
            );

            if swap_accepted {
                populations.swap(i, i + 1);
                fitnesses.swap(i, i + 1);
                constraints.swap(i, i + 1);
                step_sizes.swap(i, i + 1);
            }

            let current_success = if swap_accepted { 1.0 } else { 0.0 };
            self.swap_acceptance_rates[i] = self.swap_rate_smoothing * current_success
                + (1.0 - self.swap_rate_smoothing) * self.swap_acceptance_rates[i];
        }

        if self.adaptive_swapping && self.rng.random::<f64>() < self.random_swap_probability {
            self.attempt_random_swap(
                populations,
                fitnesses,
                constraints,
                step_sizes,
                temperatures,
            );
        }
    }

    /// Attempt random non-adjacent swaps
    fn attempt_random_swap(
        &mut self,
        populations: &mut [OMatrix<T, N, D>],
        fitnesses: &mut [OVector<T, N>],
        constraints: &mut [OVector<bool, N>],
        step_sizes: &mut [Vec<OMatrix<T, D, D>>],
        temperatures: &[T],
    ) {
        let n = populations.len();
        if n < 3 {
            return;
        }

        let i = self.rng.random_range(0..n);
        let mut j = self.rng.random_range(0..n);

        while j == i || j == i.wrapping_sub(1) || j == i + 1 {
            j = self.rng.random_range(0..n);
        }

        let t_i = temperatures[i];
        let t_j = temperatures[j];

        if self.metropolis_hastings.accept_replica_exchange::<N>(
            &fitnesses[i],
            &fitnesses[j],
            t_i,
            t_j,
        ) {
            populations.swap(i, j);
            fitnesses.swap(i, j);
            constraints.swap(i, j);
            step_sizes.swap(i, j);
        }
    }

    pub fn get_swap_acceptance_rates(&self) -> &[f64] {
        &self.swap_acceptance_rates
    }
}

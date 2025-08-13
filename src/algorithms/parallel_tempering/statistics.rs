use crate::utils::opt_prob::FloatNumber as FloatNum;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DimSub, OMatrix, OVector, RealField};

pub struct Statistics<T, N, D>
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
    _phantom: std::marker::PhantomData<(T, N, D)>,
}

impl<T, N, D> Statistics<T, N, D>
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
    pub fn compute_effective_sample_size(fitnesses: &[OVector<T, N>]) -> Vec<f64> {
        let mut ess_values = Vec::with_capacity(fitnesses.len());

        for fitness in fitnesses {
            let fitness_chain: Vec<f64> =
                fitness.iter().map(|f| f.to_f64().unwrap_or(0.0)).collect();

            let ess = Self::compute_ess_for_chain(&fitness_chain);
            ess_values.push(ess);
        }

        ess_values
    }

    /// ESS using autocorrelation function
    fn compute_ess_for_chain(chain: &[f64]) -> f64 {
        let n = chain.len();
        if n < 10 {
            return n as f64;
        }

        let mean = chain.iter().sum::<f64>() / n as f64;
        let variance = chain.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

        if variance < 1e-12 {
            return n as f64; // Constant chain, perfect mixing
        }

        let max_lag = (n / 4).min(200);
        let mut autocorr = Vec::with_capacity(max_lag);

        for lag in 0..max_lag {
            let mut sum = 0.0;
            let count = n - lag;

            for i in 0..count {
                sum += (chain[i] - mean) * (chain[i + lag] - mean);
            }

            let rho = sum / (count as f64 * variance);
            autocorr.push(rho);

            if lag > 5 && rho < 0.01 {
                break;
            }
        }

        // Integrated autocorrelation time: τ = 1 + 2 * Σ ρ(k)
        let mut tau_int = 1.0;
        let mut cumsum = 0.0;

        for (k, &rho) in autocorr.iter().enumerate().skip(1) {
            if rho <= 0.0 {
                break;
            }

            cumsum += rho;
            let current_tau = 1.0 + 2.0 * cumsum;

            if k as f64 >= 6.0 * current_tau {
                break;
            }

            tau_int = current_tau;
        }

        let ess = n as f64 / (2.0 * tau_int + 1.0);

        ess.min(n as f64).max(1.0)
    }

    /// Population diversity across replicas
    pub fn compute_population_diversity(
        fitnesses: &[OVector<T, N>],
        constraints: &[OVector<bool, N>],
        num_replicas: usize,
    ) -> f64 {
        let mut total_variance = 0.0;

        if num_replicas < 2 {
            return 0.0;
        }

        let replica_best_fitness: Vec<f64> = (0..num_replicas)
            .map(|i| {
                fitnesses[i]
                    .iter()
                    .zip(constraints[i].iter())
                    .filter_map(|(f, c)| if *c { Some(f.to_f64().unwrap()) } else { None })
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();

        let mean_fitness = replica_best_fitness.iter().sum::<f64>() / num_replicas as f64;

        for fitness in replica_best_fitness {
            total_variance += (fitness - mean_fitness).powi(2);
        }

        total_variance / num_replicas as f64
    }
}

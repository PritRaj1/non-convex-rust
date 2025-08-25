use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OVector, U1};

use crate::utils::config::CMAESConf;
use crate::utils::opt_prob::FloatNumber as FloatNum;

#[derive(Clone)]
pub struct Parameters<T>
where
    T: FloatNum,
    OVector<T, Dyn>: Send + Sync,
    DefaultAllocator: Allocator<Dyn>,
{
    pub weights: OVector<T, Dyn>,
    pub weights_negative: Option<OVector<T, Dyn>>,
    pub mu: usize,
    pub lambda: usize,
    pub mu_neg: usize,
    pub mueff: T,
    pub mueff_neg: T,
    pub cc: T,
    pub cs: T,
    pub c1: T,
    pub cmu: T,
    pub cmu_neg: T,
    pub damps: T,
    pub chi_n: T,
}

impl<T: FloatNum> Parameters<T> {
    pub fn new<D: Dim>(conf: &CMAESConf, init_x: &OVector<T, D>, pop_size: usize) -> Self
    where
        T: Send + Sync,
        OVector<T, D>: Send + Sync,
        OVector<T, Dyn>: Send + Sync,
        DefaultAllocator: Allocator<Dyn> + Allocator<D>,
    {
        let n = init_x.len();
        let lambda = pop_size;
        let mu = conf.num_parents;

        let weights = Self::compute_weights(mu, lambda);
        let mueff = T::one() / weights.map(|w| w * w).sum();
        let n_f = T::from_f64(n as f64).unwrap();

        let (weights_negative, mu_neg, mueff_neg, cmu_neg) = if conf.use_active_cma {
            let mu_neg = (T::from_f64(conf.active_cma_ratio).unwrap()
                * T::from_f64(lambda as f64).unwrap())
            .floor()
            .to_usize()
            .unwrap()
            .min(lambda - mu);
            if mu_neg > 0 {
                let weights_neg = Self::compute_negative_weights(mu, mu_neg, lambda);
                let mueff_neg = T::one() / weights_neg.map(|w| w * w).sum();
                let cmu_neg = Self::compute_negative_learning_rate(mueff_neg, n_f);
                (Some(weights_neg), mu_neg, mueff_neg, cmu_neg)
            } else {
                (None, 0, T::zero(), T::zero())
            }
        } else {
            (None, 0, T::zero(), T::zero())
        };

        let (cc, cs) = Self::compute_time_constants(mueff, n_f);
        let (c1, cmu) = Self::compute_learning_rates(mueff, n_f);
        let damps = Self::compute_damping(mueff, n_f, cs);
        let chi_n = Self::compute_chi_n(n_f);

        Self {
            weights,
            weights_negative,
            mu,
            lambda,
            mu_neg,
            mueff,
            mueff_neg,
            cc,
            cs,
            c1,
            cmu,
            cmu_neg,
            damps,
            chi_n,
        }
    }

    // Log-linear ranked weights
    fn compute_weights(mu: usize, lambda: usize) -> OVector<T, Dyn>
    where
        DefaultAllocator: Allocator<Dyn>,
    {
        let mut weights: OVector<T, Dyn> =
            OVector::from_element_generic(Dyn::from_usize(mu), U1, T::one());
        for i in 0..mu {
            weights[i] = T::ln(T::from_f64((lambda as f64 + 1.0) / 2.0).unwrap())
                - T::ln(T::from_f64((i + 1) as f64).unwrap());
        }
        weights /= weights.sum();
        weights
    }

    fn compute_time_constants(mueff: T, n_f: T) -> (T, T) {
        let cc = (T::from_f64(4.0).unwrap() + mueff / n_f)
            / (n_f + T::from_f64(4.0).unwrap() + T::from_f64(2.0).unwrap() * mueff / n_f);

        let cs = (mueff + T::from_f64(2.0).unwrap()) / (n_f + mueff + T::from_f64(5.0).unwrap());

        (cc, cs)
    }

    fn compute_learning_rates(mueff: T, n_f: T) -> (T, T) {
        let c1 = T::from_f64(2.0).unwrap() / ((n_f + T::from_f64(1.3).unwrap()).powi(2) + mueff);

        let cmu = T::min(
            T::one() - c1,
            T::from_f64(2.0).unwrap() * (mueff - T::from_f64(2.0).unwrap() + T::one() / mueff)
                / ((n_f + T::from_f64(2.0).unwrap()).powi(2) + mueff),
        );

        (c1, cmu)
    }

    fn compute_damping(mueff: T, n_f: T, cs: T) -> T {
        T::one()
            + T::from_f64(2.0).unwrap()
                * T::max(
                    T::zero(),
                    T::sqrt((mueff - T::one()) / (n_f + T::one())) - T::one(),
                )
            + cs
    }

    fn compute_chi_n(n_f: T) -> T {
        T::sqrt(n_f)
            * (T::one() - T::one() / (T::from_f64(4.0).unwrap() * n_f)
                + T::one() / (T::from_f64(21.0).unwrap() * n_f.powi(2)))
    }

    fn compute_negative_weights(mu: usize, mu_neg: usize, lambda: usize) -> OVector<T, Dyn>
    where
        DefaultAllocator: Allocator<Dyn>,
    {
        let mut weights_neg: OVector<T, Dyn> =
            OVector::from_element_generic(Dyn::from_usize(mu_neg), U1, T::zero());

        // Starting from rank mu+1 (0-indexed: mu) to rank mu+mu_neg
        for i in 0..mu_neg {
            let rank = mu + i + 1; // 1-indexed rank
            weights_neg[i] = -T::ln(T::from_f64((lambda as f64 + 1.0) / 2.0).unwrap())
                + T::ln(T::from_f64(rank as f64).unwrap());
        }

        let sum_neg = -weights_neg.sum();
        if sum_neg > T::zero() {
            weights_neg /= sum_neg;
            weights_neg *= -T::one(); // Ensure negative
        }

        weights_neg
    }

    /// Ssmaller than the positive lr to avoid instability
    fn compute_negative_learning_rate(mueff_neg: T, n_f: T) -> T {
        let alpha_neg = T::from_f64(0.4).unwrap(); // Damping
        let cmu_neg_base = T::from_f64(2.0).unwrap() * mueff_neg
            / ((n_f + T::from_f64(2.0).unwrap()).powi(2) + mueff_neg);
        T::min(alpha_neg, cmu_neg_base)
    }
}

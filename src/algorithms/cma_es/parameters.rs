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
    pub mu: usize,
    pub lambda: usize,
    pub mueff: T,
    pub cc: T,
    pub cs: T,
    pub c1: T,
    pub cmu: T,
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

        let (cc, cs) = Self::compute_time_constants(mueff, n_f);
        let (c1, cmu) = Self::compute_learning_rates(mueff, n_f);
        let damps = Self::compute_damping(mueff, n_f, cs);
        let chi_n = Self::compute_chi_n(n_f);

        Self {
            weights,
            mu,
            lambda,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chi_n,
        }
    }

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
}

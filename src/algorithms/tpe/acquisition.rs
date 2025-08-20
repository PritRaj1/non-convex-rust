use crate::utils::opt_prob::FloatNumber as FloatNum;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use std::iter::Sum;

use crate::algorithms::tpe::kernels::KernelDensityEstimator;

pub struct ExpectedImprovement<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    _phantom: std::marker::PhantomData<D>,
}

impl<D> ExpectedImprovement<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn compute_ei<T: FloatNum + Sum>(
        &self,
        x: &OVector<T, D>,
        kde_l: &KernelDensityEstimator<T, D>,
        kde_g: &KernelDensityEstimator<T, D>,
        prior_weight: T,
    ) -> T
    where
        OVector<T, D>: Send + Sync,
        DefaultAllocator: Allocator<D>,
    {
        let p_l = kde_l.evaluate(x); // Low-fitness dist
        let p_g = kde_g.evaluate(x); // High-fitness dist

        if p_g <= T::from_f64(1e-10).unwrap() {
            return T::zero();
        }

        // Expected Improvement: EI(x) = E[max(f(x) - f(x_best) - ξ, 0)]
        // In TPE we use the ratio, since the truu expectation is analytically difficult
        let ratio = p_l / p_g;

        // Apply prior weight and regularization
        let ei = ratio * prior_weight;
        let max_ei = T::from_f64(100.0).unwrap();
        ei.min(max_ei)
    }
}

impl<D> Default for ExpectedImprovement<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn default() -> Self {
        Self::new()
    }
}

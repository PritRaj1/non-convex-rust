use nalgebra::{allocator::Allocator, DefaultAllocator, SVector, U2};
use non_convex_opt::utils::opt_prob::{BooleanConstraintFunction, ObjectiveFunction};

#[derive(Clone)]
pub struct KBF;

impl ObjectiveFunction<f64, U2> for KBF
where
    DefaultAllocator: Allocator<U2>,
{
    fn f(&self, x: &SVector<f64, 2>) -> f64 {
        let sum_cos4: f64 = x.iter().map(|&xi| xi.cos().powi(4)).sum();
        let prod_cos2: f64 = x.iter().map(|&xi| xi.cos().powi(2)).product();
        let sum_ix2: f64 = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (i as f64 + 1.0) * xi * xi)
            .sum();

        (sum_cos4 - 2.0 * prod_cos2).abs() / sum_ix2.sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct KBFConstraints;

impl BooleanConstraintFunction<f64, U2> for KBFConstraints
where
    DefaultAllocator: Allocator<U2>,
{
    fn g(&self, x: &SVector<f64, 2>) -> bool {
        let n = x.len();
        let product: f64 = x.iter().product();
        let sum: f64 = x.iter().sum();

        x.iter().all(|&xi| xi >= 0.0 && xi <= 10.0)
            && product > 0.75
            && sum < (15.0 * n as f64) / 2.0
    }
}

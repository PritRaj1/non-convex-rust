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

#[derive(Clone)]
pub struct MultiModalFunction;

impl ObjectiveFunction<f64, U2> for MultiModalFunction
where
    DefaultAllocator: Allocator<U2>,
{
    fn f(&self, x: &SVector<f64, 2>) -> f64 {
        let gaussian1 = -0.5 * ((x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2));
        let gaussian2 = -0.3 * ((x[0] - 7.0).powi(2) + (x[1] - 7.0).powi(2));
        let gaussian3 = -0.2 * ((x[0] - 7.0).powi(2) + (x[1] - 3.0).powi(2));

        10.0 * gaussian1.exp() + 5.0 * gaussian2.exp() + 5.0 * gaussian3.exp()
    }
    fn gradient(&self, x: &SVector<f64, 2>) -> Option<SVector<f64, 2>> {
        let mut grad = SVector::<f64, 2>::zeros();

        let exp1 = (-0.5 * ((x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2))).exp();
        grad[0] += 10.0 * exp1 * (-2.0 * 0.5 * (x[0] - 3.0));
        grad[1] += 10.0 * exp1 * (-2.0 * 0.5 * (x[1] - 3.0));

        let exp2 = (-0.3 * ((x[0] - 7.0).powi(2) + (x[1] - 7.0).powi(2))).exp();
        grad[0] += 5.0 * exp2 * (-2.0 * 0.3 * (x[0] - 7.0));
        grad[1] += 5.0 * exp2 * (-2.0 * 0.3 * (x[1] - 7.0));

        let exp3 = (-0.2 * ((x[0] - 7.0).powi(2) + (x[1] - 3.0).powi(2))).exp();
        grad[0] += 5.0 * exp3 * (-2.0 * 0.2 * (x[0] - 7.0));
        grad[1] += 5.0 * exp3 * (-2.0 * 0.2 * (x[1] - 3.0));

        Some(grad)
    }
}

#[derive(Debug, Clone)]
pub struct BoxConstraints;

impl BooleanConstraintFunction<f64, U2> for BoxConstraints
where
    DefaultAllocator: Allocator<U2>,
{
    fn g(&self, x: &SVector<f64, 2>) -> bool {
        x.iter().all(|&xi| xi >= 0.0 && xi <= 10.0)
    }
}

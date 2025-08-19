use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use non_convex_opt::utils::opt_prob::{BooleanConstraintFunction, ObjectiveFunction};

#[allow(dead_code)]
#[derive(Clone)]
pub struct Kbf;

impl<D> ObjectiveFunction<f64, D> for Kbf
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn f(&self, x: &OVector<f64, D>) -> f64 {
        let sum_cos4: f64 = x.iter().map(|&xi| xi.cos().powi(4)).sum();
        let prod_cos2: f64 = x.iter().map(|&xi| xi.cos().powi(2)).product();
        let sum_ix2: f64 = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (i as f64 + 1.0) * xi * xi)
            .sum();
        (sum_cos4 - 2.0 * prod_cos2).abs() / sum_ix2.sqrt()
    }

    fn x_lower_bound(&self, x: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        Some(OVector::<f64, D>::zeros_generic(D::from_usize(x.len()), U1))
    }

    fn x_upper_bound(&self, x: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        Some(OVector::<f64, D>::from_element_generic(
            D::from_usize(x.len()),
            U1,
            10.0,
        ))
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct KbfConstraints;

impl<D: Dim> BooleanConstraintFunction<f64, D> for KbfConstraints
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn g(&self, x: &OVector<f64, D>) -> bool {
        let n = x.len();
        let product: f64 = x.iter().product();
        let sum: f64 = x.iter().sum();

        x.iter().all(|&xi| (-5.0..=5.0).contains(&xi))
            && product > 0.75
            && sum < (15.0 * n as f64) / 2.0
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct MultiModalFunction;

impl<D: Dim> ObjectiveFunction<f64, D> for MultiModalFunction
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn f(&self, x: &OVector<f64, D>) -> f64 {
        let n = x.len();
        let mut sum = 0.0;
        for i in 0..n {
            sum += x[i].sin() * x[i].cos() + 0.1 * x[i].powi(2);
        }
        sum
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct BoxConstraints;

impl<D: Dim> BooleanConstraintFunction<f64, D> for BoxConstraints
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn g(&self, x: &OVector<f64, D>) -> bool {
        x.iter().all(|&xi| (-5.0..=5.0).contains(&xi))
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct RosenbrockFunction;

impl<D: Dim> ObjectiveFunction<f64, D> for RosenbrockFunction
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn f(&self, x: &OVector<f64, D>) -> f64 {
        if x.len() < 2 {
            return f64::NEG_INFINITY;
        }
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    fn gradient(&self, x: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        // Catch empty vector
        if x.len() < 2 {
            return Some(OVector::<f64, D>::zeros_generic(D::from_usize(x.len()), U1));
        }

        let mut grad = OVector::<f64, D>::zeros_generic(D::from_usize(x.len()), U1);
        grad[0] = -400.0 * x[0] * (x[1] - x[0].powi(2)) - 2.0 * (1.0 - x[0]);

        for i in 1..x.len() - 1 {
            grad[i] = 200.0 * (x[i] - x[i - 1].powi(2))
                - 400.0 * x[i] * (x[i + 1] - x[i].powi(2))
                - 2.0 * (1.0 - x[i]);
        }

        let n = x.len() - 1;
        grad[n] = 200.0 * (x[n] - x[n - 1].powi(2));
        Some(grad)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RosenbrockConstraints;

impl<D> BooleanConstraintFunction<f64, D> for RosenbrockConstraints
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn g(&self, x: &OVector<f64, D>) -> bool {
        x.iter().all(|&xi| (0.0..=1.0).contains(&xi))
    }
}

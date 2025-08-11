use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use non_convex_opt::utils::opt_prob::{BooleanConstraintFunction, ObjectiveFunction};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RosenbrockObjective {
    pub a: f64,
    pub b: f64,
}

impl<D: Dim> ObjectiveFunction<f64, D> for RosenbrockObjective
where
    DefaultAllocator: Allocator<D>,
{
    fn f(&self, x: &OVector<f64, D>) -> f64 {
        let n = x.len();
        let mut sum = 0.0;
        for i in 0..n - 1 {
            sum += self.b * (x[i + 1] - x[i].powi(2)).powi(2) + (self.a - x[i]).powi(2);
        }
        sum
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RosenbrockConstraints {}

impl<D: Dim> BooleanConstraintFunction<f64, D> for RosenbrockConstraints
where
    DefaultAllocator: Allocator<D>,
{
    fn g(&self, x: &OVector<f64, D>) -> bool {
        x.iter().all(|&xi| xi >= 0.0 && xi <= 1.0)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct QuadraticObjective {
    pub a: f64,
    pub b: f64,
}

impl<D: Dim> ObjectiveFunction<f64, D> for QuadraticObjective
where
    DefaultAllocator: Allocator<D>,
{
    fn f(&self, x: &OVector<f64, D>) -> f64 {
        let n = x.len();
        let mut sum = 0.0;
        for i in 0..n {
            sum -= self.b * x[i].powi(2) - self.a * x[i];
        }
        sum
    }

    fn gradient(&self, x: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        let n = x.len();
        let mut grad = OVector::<f64, D>::zeros_generic(D::from_usize(n), U1);
        for i in 0..n {
            grad[i] = -2.0 * self.b * x[i] + self.a;
        }
        Some(grad)
    }

    fn x_lower_bound(&self, x: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        Some(OVector::<f64, D>::from_element_generic(
            D::from_usize(x.len()),
            U1,
            0.0,
        ))
    }

    fn x_upper_bound(&self, x: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        Some(OVector::<f64, D>::from_element_generic(
            D::from_usize(x.len()),
            U1,
            1.0,
        ))
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct QuadraticConstraints {}

impl<D: Dim> BooleanConstraintFunction<f64, D> for QuadraticConstraints
where
    DefaultAllocator: Allocator<D>,
{
    fn g(&self, x: &OVector<f64, D>) -> bool {
        x.iter().all(|&xi| xi >= 0.0 && xi <= 1.0)
    }
}

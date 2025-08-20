use crate::utils::opt_prob::FloatNumber as FloatNum;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use rayon::prelude::*;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum KernelType {
    Gaussian,
    Epanechnikov,
    TopHat,
    Triangular,
}

pub trait Kernel<T: FloatNum>: Send + Sync {
    fn evaluate(&self, x: T, bandwidth: T) -> T;
}

/// RBF: K(x) = (1/√(2π)) * exp(-x²/2)
pub struct GaussianKernel;

impl<T: FloatNum> Kernel<T> for GaussianKernel {
    fn evaluate(&self, x: T, bandwidth: T) -> T {
        let x_norm = x / bandwidth;
        let factor = T::from_f64(0.3989422804014327).unwrap(); // 1/√(2π)
        factor * (-x_norm * x_norm / T::from_f64(2.0).unwrap()).exp() / bandwidth
    }
}

/// Epanechnikov: K(x) = (3/4) * (1 - x²) for |x| ≤ 1, 0 otherwise
pub struct EpanechnikovKernel;

impl<T: FloatNum> Kernel<T> for EpanechnikovKernel {
    fn evaluate(&self, x: T, bandwidth: T) -> T {
        let x_norm = x / bandwidth;
        let x_abs = x_norm.abs();

        if x_abs <= T::one() {
            let factor = T::from_f64(0.75).unwrap(); // 3/4
            factor * (T::one() - x_norm * x_norm) / bandwidth
        } else {
            T::zero()
        }
    }
}

/// Uniform: K(x) = 1/2 for |x| ≤ 1, 0 otherwise
pub struct TopHatKernel;

impl<T: FloatNum> Kernel<T> for TopHatKernel {
    fn evaluate(&self, x: T, bandwidth: T) -> T {
        let x_norm = x / bandwidth;

        if x_norm.abs() <= T::one() {
            T::from_f64(0.5).unwrap() / bandwidth
        } else {
            T::zero()
        }
    }
}

/// Triangular: K(x) = (1 - |x|) for |x| ≤ 1, 0 otherwise
pub struct TriangularKernel;

impl<T: FloatNum> Kernel<T> for TriangularKernel {
    fn evaluate(&self, x: T, bandwidth: T) -> T {
        let x_norm = x / bandwidth;
        let x_abs = x_norm.abs();

        if x_abs <= T::one() {
            (T::one() - x_abs) / bandwidth
        } else {
            T::zero()
        }
    }
}

// Builder
pub fn create_kernel<T: FloatNum>(kernel_type: KernelType) -> Box<dyn Kernel<T>> {
    match kernel_type {
        KernelType::Gaussian => Box::new(GaussianKernel),
        KernelType::Epanechnikov => Box::new(EpanechnikovKernel),
        KernelType::TopHat => Box::new(TopHatKernel),
        KernelType::Triangular => Box::new(TriangularKernel),
    }
}

pub struct KernelDensityEstimator<T: FloatNum, D: Dim>
where
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    data: Vec<OVector<T, D>>,
    bandwidths: OVector<T, D>,
    kernel: Box<dyn Kernel<T>>,
    _phantom: PhantomData<T>,
}

impl<T: FloatNum + std::iter::Sum, D: Dim> KernelDensityEstimator<T, D>
where
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    pub fn new(kernel_type: KernelType, dimension_size: usize) -> Self {
        Self {
            data: Vec::new(),
            bandwidths: OVector::zeros_generic(D::from_usize(dimension_size), U1),
            kernel: create_kernel(kernel_type),
            _phantom: PhantomData,
        }
    }

    pub fn fit(&mut self, data: &[OVector<T, D>]) {
        self.data = data.to_vec();
        self.bandwidths = self.compute_bandwidths();
    }

    pub fn evaluate(&self, x: &OVector<T, D>) -> T {
        if self.data.is_empty() {
            return T::zero();
        }

        let n = T::from_usize(self.data.len()).unwrap();

        let density: T = self
            .data
            .par_iter()
            .map(|point| {
                let mut kernel_product = T::one();

                for dim in 0..x.len() {
                    let diff = x[dim] - point[dim];
                    let bandwidth = self.bandwidths[dim];
                    kernel_product *= self.kernel.evaluate(diff, bandwidth);
                }

                kernel_product
            })
            .sum();

        density / n
    }

    fn compute_bandwidths(&self) -> OVector<T, D> {
        if self.data.is_empty() {
            return self.bandwidths.clone();
        }

        let mut bandwidths = self.bandwidths.clone();
        let n = self.data.len();

        // Silverman's rule of thumb: h = 1.06 * σ * n^(-1/5)
        for dim in 0..self.data[0].len() {
            let variance = self.compute_variance(dim);
            let std_dev = variance.sqrt();
            let factor = T::from_f64(1.06).unwrap();
            let n_factor = T::from_f64(n as f64)
                .unwrap()
                .powf(T::from_f64(-0.2).unwrap());

            bandwidths[dim] = factor * std_dev * n_factor;
            let min_bw = T::from_f64(1e-6).unwrap(); // Avoid zero bandwidth
            bandwidths[dim] = bandwidths[dim].max(min_bw);
        }

        bandwidths
    }

    fn compute_variance(&self, dim: usize) -> T {
        if self.data.len() < 2 {
            return T::zero();
        }

        let mean = self.data.par_iter().map(|point| point[dim]).sum::<T>()
            / T::from_usize(self.data.len()).unwrap();

        let variance = self
            .data
            .par_iter()
            .map(|point| {
                let diff = point[dim] - mean;
                diff * diff
            })
            .sum::<T>()
            / T::from_usize(self.data.len() - 1).unwrap();

        variance
    }
}

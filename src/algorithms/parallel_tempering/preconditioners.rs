use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1, RealField};


use std::iter::Sum;

use crate::utils::opt_prob::FloatNumber as FloatNum;

pub trait Preconditioner<T, N, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<D, D> + Allocator<N>,
{
    fn compute_covariance(
        &self,
        population: &OMatrix<T, N, D>,
        fitness: &OVector<T, N>,
        constraints: &OVector<bool, N>,
    ) -> OMatrix<T, D, D>;

    fn name(&self) -> &'static str;
}

/// Uses empirical covariance of the population
pub struct SampleCovariance<T, D>
where
    T: FloatNum + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    regularization: T,
    _phantom: std::marker::PhantomData<D>,
}

impl<T, D> SampleCovariance<T, D>
where
    T: FloatNum + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    pub fn new(regularization: T) -> Self {
        Self {
            regularization,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, N, D> Preconditioner<T, N, D> for SampleCovariance<T, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<D, D> + Allocator<U1, D> + Allocator<N>,
{
    fn compute_covariance(
        &self,
        population: &OMatrix<T, N, D>,
        _fitness: &OVector<T, N>,
        constraints: &OVector<bool, N>,
    ) -> OMatrix<T, D, D> {
        let dim = population.ncols();
        let mut valid_individuals = Vec::new();

        for i in 0..population.nrows() {
            if constraints[i] {
                valid_individuals.push(population.row(i).transpose());
            }
        }

        // Return identity matrix if no feasible individuals
        if valid_individuals.is_empty() {
            return OMatrix::<T, D, D>::identity_generic(
                D::from_usize(dim),
                D::from_usize(dim),
            ) * self.regularization;
        }

        let mut mean = OVector::<T, D>::zeros_generic(D::from_usize(dim), U1);
        for individual in &valid_individuals {
            mean += individual;
        }
        mean /= T::from_usize(valid_individuals.len()).unwrap();

        let mut cov = OMatrix::<T, D, D>::zeros_generic(
            D::from_usize(dim),
            D::from_usize(dim),
        );

        for individual in &valid_individuals {
            let diff = individual - &mean;
            cov += &diff * diff.transpose();
        }

        let n = T::from_usize(valid_individuals.len().saturating_sub(1).max(1)).unwrap();
        cov /= n;
        
        for i in 0..dim {
            cov[(i, i)] += self.regularization;
        }

        cov
    }

    fn name(&self) -> &'static str {
        "SampleCovariance"
    }
}

/// Weights individuals by their fitness
pub struct FitnessWeightedCovariance<T, D>
where
    T: FloatNum + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    regularization: T,
    elite_fraction: T, // Fraction of top individuals to consider
    _phantom: std::marker::PhantomData<D>,
}

impl<T, D> FitnessWeightedCovariance<T, D>
where
    T: FloatNum + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    pub fn new(regularization: T, elite_fraction: T) -> Self {
        Self {
            regularization,
            elite_fraction: RealField::max(RealField::min(elite_fraction, T::from_f64(1.0).unwrap()), T::from_f64(0.1).unwrap()),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, N, D> Preconditioner<T, N, D> for FitnessWeightedCovariance<T, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<D, D> + Allocator<U1, D> + Allocator<N>,
    OVector<T, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
{
    fn compute_covariance(
        &self,
        population: &OMatrix<T, N, D>,
        fitness: &OVector<T, N>,
        constraints: &OVector<bool, N>,
    ) -> OMatrix<T, D, D> {
        let dim = population.ncols();
        let mut valid_data: Vec<(OVector<T, D>, T)> = Vec::new();

        // Collect feasible individuals with their fitness
        for i in 0..population.nrows() {
            if constraints[i] {
                valid_data.push((population.row(i).transpose(), fitness[i]));
            }
        }

        if valid_data.is_empty() {
            return OMatrix::<T, D, D>::identity_generic(
                D::from_usize(dim),
                D::from_usize(dim),
            ) * self.regularization;
        }

        valid_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take elite fraction from sorted pop
        let elite_count = ((valid_data.len() as f64 * self.elite_fraction.to_f64().unwrap()).ceil() as usize).max(1);
        let elite_data: Vec<_> = valid_data.into_iter().take(elite_count).collect();

        let total_fitness: T = elite_data.iter().map(|(_, f)| *f).sum();
        let mut weighted_mean = OVector::<T, D>::zeros_generic(D::from_usize(dim), U1);
        
        for (individual, fitness) in &elite_data {
            let weight = *fitness / total_fitness;
            weighted_mean += individual * weight;
        }

        let mut cov = OMatrix::<T, D, D>::zeros_generic(
            D::from_usize(dim),
            D::from_usize(dim),
        );

        for (individual, fitness) in &elite_data {
            let weight = *fitness / total_fitness;
            let diff = individual - &weighted_mean;
            cov += &diff * diff.transpose() * weight;
        }

        for i in 0..dim {
            cov[(i, i)] += self.regularization;
        }

        cov
    }

    fn name(&self) -> &'static str {
        "FitnessWeightedCovariance"
    }
}

/// Adapts based on acceptance rates and iteration
pub struct AdaptiveCovariance<T, D>
where
    T: FloatNum + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    regularization: T,
    adaptation_rate: T,
    target_acceptance_rate: T,
    min_regularization: T,
    max_regularization: T,
    _phantom: std::marker::PhantomData<D>,
}

impl<T, D> AdaptiveCovariance<T, D>
where
    T: FloatNum + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    pub fn new(
        initial_regularization: T,
        adaptation_rate: T,
        target_acceptance_rate: T,
    ) -> Self {
        Self {
            regularization: initial_regularization,
            adaptation_rate,
            target_acceptance_rate,
            min_regularization: initial_regularization * T::from_f64(0.01).unwrap(),
            max_regularization: initial_regularization * T::from_f64(100.0).unwrap(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn update_regularization(&mut self, acceptance_rate: T) {
        let rate_diff = acceptance_rate - self.target_acceptance_rate;
        let adaptation = T::from_f64(1.0).unwrap() + self.adaptation_rate * rate_diff;
        
        self.regularization *= adaptation;
        self.regularization = RealField::min(RealField::max(self.regularization, self.min_regularization), self.max_regularization);
    }
}

impl<T, N, D> Preconditioner<T, N, D> for AdaptiveCovariance<T, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<D, D> + Allocator<U1, D> + Allocator<N>,
    OVector<T, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
{
    fn compute_covariance(
        &self,
        population: &OMatrix<T, N, D>,
        fitness: &OVector<T, N>,
        constraints: &OVector<bool, N>,
    ) -> OMatrix<T, D, D> {
        let sample_preconditioner = SampleCovariance::new(self.regularization);
        sample_preconditioner.compute_covariance(population, fitness, constraints)
    }

    fn name(&self) -> &'static str {
        "AdaptiveCovariance"
    }
}

/// Uses Ledoit-Wolf shrinkage estimation
pub struct ShrinkageCovariance<T, D>
where
    T: FloatNum + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    shrinkage_intensity: T,
    _phantom: std::marker::PhantomData<D>,
}

impl<T, D> ShrinkageCovariance<T, D>
where
    T: FloatNum + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    pub fn new(shrinkage_intensity: T) -> Self {
        Self {
            shrinkage_intensity: RealField::max(RealField::min(shrinkage_intensity, T::from_f64(1.0).unwrap()), T::from_f64(0.0).unwrap()),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, N, D> Preconditioner<T, N, D> for ShrinkageCovariance<T, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<D, D> + Allocator<U1, D> + Allocator<N>,
    OVector<T, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
{
    fn compute_covariance(
        &self,
        population: &OMatrix<T, N, D>,
        fitness: &OVector<T, N>,
        constraints: &OVector<bool, N>,
    ) -> OMatrix<T, D, D> {
        let dim = population.ncols();
        
        let sample_preconditioner = SampleCovariance::new(T::from_f64(1e-6).unwrap());
        let sample_cov = sample_preconditioner.compute_covariance(population, fitness, constraints);
        
        let trace = (0..dim).map(|i| sample_cov[(i, i)]).sum::<T>();
        let target_variance = trace / T::from_usize(dim).unwrap();
        
        let identity = OMatrix::<T, D, D>::identity_generic(
            D::from_usize(dim),
            D::from_usize(dim),
        ) * target_variance;
        
        let one_minus_shrinkage = T::from_f64(1.0).unwrap() - self.shrinkage_intensity;
        sample_cov * one_minus_shrinkage + identity * self.shrinkage_intensity
    }

    fn name(&self) -> &'static str {
        "ShrinkageCovariance"
    }
}
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, DimSub, OMatrix, OVector, RealField, U1,
};
use rayon::prelude::*;

/// Manages the state of a single replica in parallel tempering
pub struct ReplicaState<T, N, D>
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
        + Allocator<U1, D>
        + Allocator<<D as DimSub<nalgebra::Const<1>>>::Output>,
{
    pub population: OMatrix<T, N, D>,
    pub fitness: OVector<T, N>,
    pub constraints: OVector<bool, N>,
    pub step_sizes: Vec<OMatrix<T, D, D>>,
    pub acceptance_rate: f64,
}

impl<T, N, D> ReplicaState<T, N, D>
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
        + Allocator<U1, D>
        + Allocator<<D as DimSub<nalgebra::Const<1>>>::Output>,
{
    pub fn new(
        init_pop: &OMatrix<T, N, D>,
        opt_prob: &OptProb<T, D>,
        step_size_value: f64,
    ) -> Self {
        let mut population = OMatrix::<T, N, D>::zeros_generic(
            N::from_usize(init_pop.nrows()),
            D::from_usize(init_pop.ncols()),
        );
        for i in 0..init_pop.nrows() {
            population.set_row(i, &init_pop.row(i));
        }

        let fitness: Vec<T> = (0..init_pop.nrows())
            .into_par_iter()
            .map(|i| {
                let individual = init_pop.row(i).transpose();
                opt_prob.evaluate(&individual)
            })
            .collect();

        let constraints: Vec<bool> = (0..init_pop.nrows())
            .into_par_iter()
            .map(|i| {
                let individual = init_pop.row(i).transpose();
                opt_prob.is_feasible(&individual)
            })
            .collect();

        let step_sizes: Vec<OMatrix<T, D, D>> = (0..population.nrows())
            .map(|_| {
                OMatrix::<T, D, D>::identity_generic(
                    D::from_usize(population.ncols()),
                    D::from_usize(population.ncols()),
                ) * T::from_f64(step_size_value).unwrap()
            })
            .collect();

        Self {
            population,
            fitness: OVector::<T, N>::from_vec_generic(
                N::from_usize(init_pop.nrows()),
                U1,
                fitness,
            ),
            constraints: OVector::<bool, N>::from_vec_generic(
                N::from_usize(init_pop.nrows()),
                U1,
                constraints,
            ),
            step_sizes,
            acceptance_rate: 0.5,
        }
    }

    pub fn find_best_individual(&self) -> Option<(OVector<T, D>, T)> {
        let mut best_idx = None;
        let mut best_fitness = T::from_f64(f64::NEG_INFINITY).unwrap();

        for i in 0..self.fitness.len() {
            if self.constraints[i] && self.fitness[i] > best_fitness {
                best_fitness = self.fitness[i];
                best_idx = Some(i);
            }
        }

        best_idx.map(|idx| {
            let best_individual = self.population.row(idx).transpose().into_owned();
            (best_individual, best_fitness)
        })
    }

    /// Update acceptance rate with exponential moving average
    pub fn update_acceptance_rate(&mut self, current_rate: f64, smoothing: f64) {
        self.acceptance_rate = smoothing * current_rate + (1.0 - smoothing) * self.acceptance_rate;
    }

    pub fn num_individuals(&self) -> usize {
        self.population.nrows()
    }

    pub fn dimensionality(&self) -> usize {
        self.population.ncols()
    }
}

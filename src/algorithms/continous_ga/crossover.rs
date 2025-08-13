use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1};
use rand::Rng;
use rayon::prelude::*;

use crate::utils::opt_prob::FloatNumber as FloatNum;

pub trait CrossoverOperator<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, N>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<Dyn, D> + Allocator<N, D> + Allocator<N>,
{
    fn crossover(&self, parents: &OMatrix<T, Dyn, D>) -> OMatrix<T, N, D>;
}

pub struct Random {
    pub crossover_prob: f64, // F64 for RNG
    pub population_size: usize,
}

impl Random {
    pub fn new(crossover_prob: f64, population_size: usize) -> Self {
        Self {
            crossover_prob,
            population_size,
        }
    }
}

impl<T: FloatNum, N: Dim, D: Dim> CrossoverOperator<T, N, D> for Random
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<Dyn, D> + Allocator<N, D> + Allocator<N> + Allocator<D> + Allocator<U1, D>,
{
    fn crossover(&self, parents: &OMatrix<T, Dyn, D>) -> OMatrix<T, N, D> {
        let num_parents = parents.nrows();
        let crossover_prob = self.crossover_prob;
        
        let offspring_rows: Vec<_> = (0..self.population_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                let i = rng.random_range(0..num_parents);
                let mut j = rng.random_range(0..num_parents);
                
                while j == i && num_parents > 1 {
                    j = rng.random_range(0..num_parents);
                }

                if rng.random::<f64>() < crossover_prob {
                    let parent1 = parents.row(i);
                    let parent2 = parents.row(j);
                    let alpha = T::from_f64(rng.random::<f64>()).unwrap();
                    
                    let mut child = OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    for k in 0..parents.ncols() {
                        child[k] = alpha * parent1[k] + (T::one() - alpha) * parent2[k];
                    }
                    child
                } else {
                    let mut child = OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    let parent_row = parents.row(i);
                    for k in 0..parents.ncols() {
                        child[k] = parent_row[k];
                    }
                    child
                }
            })
            .collect();

        let mut offspring = OMatrix::<T, N, D>::zeros_generic(
            N::from_usize(self.population_size),
            D::from_usize(parents.ncols()),
        );
        
        for (i, child) in offspring_rows.into_iter().enumerate() {
            offspring.set_row(i, &child.transpose());
        }

        offspring
    }
}

pub struct Heuristic {
    pub crossover_prob: f64, // F64 for RNG
    pub population_size: usize,
}

impl Heuristic {
    pub fn new(crossover_prob: f64, population_size: usize) -> Self {
        Self {
            crossover_prob,
            population_size,
        }
    }
}

impl<T, N, D> CrossoverOperator<T, N, D> for Heuristic
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<Dyn, D> + Allocator<N, D> + Allocator<N> + Allocator<D> + Allocator<U1, D>,
{
    fn crossover(&self, parents: &OMatrix<T, Dyn, D>) -> OMatrix<T, N, D> {
        let num_parents = parents.nrows();
        let crossover_prob = self.crossover_prob;
        
        let offspring_rows: Vec<_> = (0..self.population_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                let i = rng.random_range(0..num_parents);
                let mut j = rng.random_range(0..num_parents);
                
                while j == i && num_parents > 1 {
                    j = rng.random_range(0..num_parents);
                }

                if rng.random::<f64>() < crossover_prob {
                    let parent1 = parents.row(i);
                    let parent2 = parents.row(j);
                    let b = T::from_f64(rng.random::<f64>()).unwrap();
                    
                    let mut child = OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    for k in 0..parents.ncols() {
                        child[k] = b * (parent1[k] - parent2[k]) + parent2[k];
                    }
                    child
                } else {
                    let mut child = OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    let parent_row = parents.row(i);
                    for k in 0..parents.ncols() {
                        child[k] = parent_row[k];
                    }
                    child
                }
            })
            .collect();

        let mut offspring = OMatrix::<T, N, D>::zeros_generic(
            N::from_usize(self.population_size),
            D::from_usize(parents.ncols()),
        );
        
        for (i, child) in offspring_rows.into_iter().enumerate() {
            offspring.set_row(i, &child.transpose());
        }

        offspring
    }
}

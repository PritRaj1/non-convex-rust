use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1};
use rand::Rng;

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
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<Dyn, D> + Allocator<N, D> + Allocator<N> + Allocator<D> + Allocator<U1, D>,
{
    fn crossover(&self, parents: &OMatrix<T, Dyn, D>) -> OMatrix<T, N, D> {
        let mut rng = rand::rng();
        let mut offspring = OMatrix::<T, N, D>::zeros_generic(
            N::from_usize(self.population_size),
            D::from_usize(parents.ncols()),
        );

        let num_parents = parents.nrows();
        let mut offspring_count = 0;

        // Pre-allocate
        let mut child: OVector<T, D> = OVector::zeros_generic(D::from_usize(parents.ncols()), U1);

        while offspring_count < self.population_size {
            let i = rng.random_range(0..num_parents);
            let mut j = rng.random_range(0..num_parents);
            
            while j == i && num_parents > 1 {
                j = rng.random_range(0..num_parents);
            }

            if rng.random::<f64>() < self.crossover_prob {
                let parent1 = parents.row(i);
                let parent2 = parents.row(j);

                let alpha = T::from_f64(rng.random::<f64>()).unwrap();
                for k in 0..parents.ncols() {
                    child[k] = alpha * parent1[k] + (T::one() - alpha) * parent2[k];
                }
                offspring.set_row(offspring_count, &child.transpose());
                offspring_count += 1;
            } else if offspring_count < self.population_size {
                offspring.set_row(offspring_count, &parents.row(i));
                offspring_count += 1;
            }
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
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<Dyn, D> + Allocator<N, D> + Allocator<N> + Allocator<D> + Allocator<U1, D>,
{
    fn crossover(&self, parents: &OMatrix<T, Dyn, D>) -> OMatrix<T, N, D> {
        let mut rng = rand::rng();
        let mut offspring = OMatrix::<T, N, D>::zeros_generic(
            N::from_usize(self.population_size),
            D::from_usize(parents.ncols()),
        );

        let num_parents = parents.nrows();
        let mut offspring_count = 0;

        // Pre-allocate
        let mut child: OVector<T, D> = OVector::zeros_generic(D::from_usize(parents.ncols()), U1);

        while offspring_count < self.population_size {
            let i = rng.random_range(0..num_parents);
            let mut j = rng.random_range(0..num_parents);
            
            while j == i && num_parents > 1 {
                j = rng.random_range(0..num_parents);
            }

            if rng.random::<f64>() < self.crossover_prob {
                let parent1 = parents.row(i);
                let parent2 = parents.row(j);

                let b = T::from_f64(rng.random::<f64>()).unwrap(); // Random factor between 0 and 1
                for k in 0..parents.ncols() {
                    child[k] = b * (parent1[k] - parent2[k]) + parent2[k]; // p_new = b * (p1 - p2) + p2
                }

                offspring.set_row(offspring_count, &child.transpose());
                offspring_count += 1;
            } else if offspring_count < self.population_size {
                offspring.set_row(offspring_count, &parents.row(i));
                offspring_count += 1;
            }
        }

        offspring
    }
}

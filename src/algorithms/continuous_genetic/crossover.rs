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

                if rng.random::<f64>() <= crossover_prob && num_parents >= 2 {
                    // Select two different parents
                    let parent1_idx = rng.random_range(0..num_parents);
                    let mut parent2_idx = rng.random_range(0..num_parents);
                    while parent2_idx == parent1_idx && num_parents > 1 {
                        parent2_idx = rng.random_range(0..num_parents);
                    }

                    let parent1 = parents.row(parent1_idx);
                    let parent2 = parents.row(parent2_idx);

                    let alpha = T::from_f64(rng.random::<f64>()).unwrap();

                    let mut child =
                        OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    for k in 0..parents.ncols() {
                        child[k] = alpha * parent1[k] + (T::one() - alpha) * parent2[k];
                    }
                    child
                } else {
                    let mut child =
                        OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    let parent_row = parents.row(rng.random_range(0..num_parents));
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
    pub crossover_prob: f64,
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

impl<T: FloatNum, N: Dim, D: Dim> CrossoverOperator<T, N, D> for Heuristic
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

                if rng.random::<f64>() <= crossover_prob && num_parents >= 2 {
                    // Select two different parents
                    let parent1_idx = rng.random_range(0..num_parents);
                    let mut parent2_idx = rng.random_range(0..num_parents);
                    while parent2_idx == parent1_idx && num_parents > 1 {
                        parent2_idx = rng.random_range(0..num_parents);
                    }

                    let parent1 = parents.row(parent1_idx);
                    let parent2 = parents.row(parent2_idx);

                    let b = T::from_f64(rng.random::<f64>()).unwrap();

                    let mut child =
                        OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    for k in 0..parents.ncols() {
                        child[k] = parent1[k] + b * (parent1[k] - parent2[k]);
                    }
                    child
                } else {
                    let mut child =
                        OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    let parent_row = parents.row(rng.random_range(0..num_parents));
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

pub struct SimulatedBinary {
    pub crossover_prob: f64,
    pub eta_c: f64, // Distribution index
    pub population_size: usize,
}

impl SimulatedBinary {
    pub fn new(crossover_prob: f64, eta_c: f64, population_size: usize) -> Self {
        Self {
            crossover_prob,
            eta_c,
            population_size,
        }
    }
}

impl<T: FloatNum, N: Dim, D: Dim> CrossoverOperator<T, N, D> for SimulatedBinary
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
        let eta_c = self.eta_c;

        // Generate all offspring in parallel
        let offspring_rows: Vec<_> = (0..self.population_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();

                if rng.random::<f64>() <= crossover_prob && num_parents >= 2 {
                    // Select two different parents
                    let parent1_idx = rng.random_range(0..num_parents);
                    let mut parent2_idx = rng.random_range(0..num_parents);
                    while parent2_idx == parent1_idx && num_parents > 1 {
                        parent2_idx = rng.random_range(0..num_parents);
                    }

                    let parent1 = parents.row(parent1_idx);
                    let parent2 = parents.row(parent2_idx);

                    // SBX crossover
                    let mut child =
                        OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    for k in 0..parents.ncols() {
                        let y1 = parent1[k];
                        let y2 = parent2[k];

                        if (y1 - y2).abs() > T::from_f64(1e-14).unwrap() {
                            let u = rng.random::<f64>();
                            let beta = if u <= 0.5 {
                                (2.0 * u).powf(1.0 / (eta_c + 1.0))
                            } else {
                                (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta_c + 1.0))
                            };

                            let beta_t = T::from_f64(beta).unwrap();
                            let half = T::from_f64(0.5).unwrap();

                            // Generate two children and randomly select one
                            let c1 = half * ((T::one() + beta_t) * y1 + (T::one() - beta_t) * y2);
                            let c2 = half * ((T::one() - beta_t) * y1 + (T::one() + beta_t) * y2);

                            child[k] = if rng.random::<bool>() { c1 } else { c2 };
                        } else {
                            // Parents are identical, copy the value
                            child[k] = y1;
                        }
                    }
                    child
                } else {
                    // No crossover, copy a random parent
                    let parent_idx = rng.random_range(0..num_parents);
                    let mut child =
                        OVector::<T, D>::zeros_generic(D::from_usize(parents.ncols()), U1);
                    let parent_row = parents.row(parent_idx);
                    for k in 0..parents.ncols() {
                        child[k] = parent_row[k];
                    }
                    child
                }
            })
            .collect();

        // Assemble final offspring matrix
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

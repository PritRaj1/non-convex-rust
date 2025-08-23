use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::utils::opt_prob::FloatNumber as FloatNum;

pub trait SelectionOperator<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, N>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<N> + Allocator<N, D> + Allocator<Dyn, D>,
{
    fn select(
        &mut self,
        population: &OMatrix<T, N, D>,
        fitness: &OVector<T, N>,
        constraints: &OVector<bool, N>,
    ) -> OMatrix<T, Dyn, D>;
}

pub struct RouletteWheel {
    pub population_size: usize,
    pub num_parents: usize,
    rng: StdRng,
}

impl RouletteWheel {
    pub fn new(population_size: usize, num_parents: usize, seed: u64) -> Self {
        RouletteWheel {
            population_size,
            num_parents,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<T, N, D> SelectionOperator<T, N, D> for RouletteWheel
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<N> + Allocator<N, D> + Allocator<Dyn, D>,
{
    fn select(
        &mut self,
        population: &OMatrix<T, N, D>,
        fitness: &OVector<T, N>,
        constraints: &OVector<bool, N>,
    ) -> OMatrix<T, Dyn, D> {
        // Normalized selection probabilities only for valid individuals
        let sum = fitness
            .iter()
            .zip(constraints.iter())
            .filter(|(_, &valid)| valid)
            .fold(T::zero(), |acc, (&x, _)| acc + x);

        let mut llhoods: OVector<T, N> = OVector::zeros_generic(N::from_usize(fitness.len()), U1);
        for (j, (&fit, &valid)) in fitness.iter().zip(constraints.iter()).enumerate() {
            if valid {
                llhoods[j] = fit / sum;
            }
        }

        let mut selected = OMatrix::<T, Dyn, D>::zeros_generic(
            Dyn::from_usize(self.num_parents),
            D::from_usize(population.ncols()),
        );

        for i in 0..self.num_parents {
            let r = T::from_f64(self.rng.random_range(0.0..1.0)).unwrap();
            let mut cumsum = T::zero();
            let mut selected_individual = false;

            for j in 0..population.nrows() {
                if !constraints[j] {
                    continue; // Skip individuals that don't satisfy constraints
                }

                cumsum += llhoods[j];
                if r <= cumsum {
                    selected.set_row(i, &population.row(j));
                    selected_individual = true;
                    break;
                }
            }

            // Fallback - never called
            if !selected_individual {
                for j in 0..population.nrows() {
                    if constraints[j] {
                        selected.set_row(i, &population.row(j));
                        break;
                    }
                }
            }
        }
        selected
    }
}

pub struct Tournament {
    pub population_size: usize,
    pub num_parents: usize,
    pub tournament_size: usize,
    seed: u64,
}

impl Tournament {
    pub fn new(
        population_size: usize,
        num_parents: usize,
        tournament_size: usize,
        seed: u64,
    ) -> Self {
        Tournament {
            population_size,
            num_parents,
            tournament_size,
            seed,
        }
    }
}

impl<T, N, D> SelectionOperator<T, N, D> for Tournament
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<Dyn, D> + Allocator<N, D> + Allocator<N>,
{
    fn select(
        &mut self,
        population: &OMatrix<T, N, D>,
        fitness: &OVector<T, N>,
        constraints: &OVector<bool, N>,
    ) -> OMatrix<T, Dyn, D> {
        let mut selected = OMatrix::<T, Dyn, D>::zeros_generic(
            Dyn::from_usize(self.num_parents),
            D::from_usize(population.ncols()),
        );

        let valid_indices: Vec<usize> = (0..population.nrows())
            .filter(|&idx| constraints[idx])
            .collect();

        if valid_indices.is_empty() {
            for i in 0..self.num_parents {
                selected.set_row(i, &population.row(0));
            }
            return selected;
        }

        let selected_indices: Vec<usize> = (0..self.num_parents)
            .into_par_iter()
            .map_init(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    StdRng::seed_from_u64(self.seed + thread_id as u64)
                },
                |rng, _| {
                    let mut tournament_indices = Vec::new();

                    let effective_tournament_size = self.tournament_size.min(valid_indices.len());
                    for _ in 0..effective_tournament_size {
                        let random_idx = rng.random_range(0..valid_indices.len());
                        tournament_indices.push(valid_indices[random_idx]);
                    }

                    let mut best_idx = tournament_indices[0];
                    let mut best_fitness = fitness[best_idx];

                    for &idx in &tournament_indices[1..] {
                        if fitness[idx] > best_fitness {
                            best_idx = idx;
                            best_fitness = fitness[idx];
                        }
                    }
                    best_idx
                },
            )
            .collect();

        // Copy selected individuals to result matrix
        for (i, &best_idx) in selected_indices.iter().enumerate() {
            selected.set_row(i, &population.row(best_idx));
        }

        selected
    }
}

pub struct Residual {
    pub population_size: usize,
    pub num_parents: usize,
    rng: StdRng,
}

impl Residual {
    pub fn new(population_size: usize, num_parents: usize, seed: u64) -> Self {
        Residual {
            population_size,
            num_parents,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<T, N, D> SelectionOperator<T, N, D> for Residual
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<Dyn, D> + Allocator<N, D> + Allocator<N>,
{
    fn select(
        &mut self,
        population: &OMatrix<T, N, D>,
        fitness: &OVector<T, N>,
        constraints: &OVector<bool, N>,
    ) -> OMatrix<T, Dyn, D> {
        let mut selected = OMatrix::<T, Dyn, D>::zeros_generic(
            Dyn::from_usize(self.num_parents),
            D::from_usize(population.ncols()),
        );

        let fitness_vec: Vec<T> = (0..fitness.len()).map(|i| fitness[i]).collect();
        let constraints_vec: Vec<bool> = (0..constraints.len()).map(|i| constraints[i]).collect();

        let sum = fitness_vec
            .par_iter()
            .zip(constraints_vec.par_iter())
            .filter(|(_, &valid)| valid)
            .map(|(&x, _)| x)
            .reduce(|| T::zero(), |acc, x| acc + x);

        let scale = T::from_f64(self.num_parents as f64).unwrap();
        let mut expected_values = vec![T::zero(); fitness.len()];
        let mut residual_probabilities = vec![T::zero(); fitness.len()];
        let mut remaining_indices = Vec::new();

        for (j, (&fit, &valid)) in fitness.iter().zip(constraints.iter()).enumerate() {
            if valid {
                let expected = (fit / sum) * scale;
                let int_part = expected.floor();
                expected_values[j] = int_part;
                residual_probabilities[j] = expected - int_part;

                if residual_probabilities[j] > T::zero() {
                    remaining_indices.push(j);
                }
            }
        }

        // Deterministic selections first (integer replication)
        let mut parent_index = 0;
        for (j, &expected_val) in expected_values.iter().enumerate().take(fitness.len()) {
            for _ in 0..expected_val.to_usize().unwrap_or(0) {
                if parent_index < self.num_parents {
                    selected.set_row(parent_index, &population.row(j));
                    parent_index += 1;
                }
            }
        }

        // Stochastic remainder selections - select based on residual probabilities
        let mut remaining_spots = self.num_parents - parent_index;

        // If no remaining indices but spots left, add all valid individuals
        if remaining_indices.is_empty() && remaining_spots > 0 {
            remaining_indices = (0..population.nrows())
                .filter(|&i| constraints[i])
                .collect();
        }

        while remaining_spots > 0 && !remaining_indices.is_empty() {
            let total_residual = remaining_indices
                .iter()
                .fold(T::zero(), |acc, &i| acc + residual_probabilities[i]);

            if total_residual <= T::zero() {
                // If all residuals are zero, select randomly
                let idx = remaining_indices[self.rng.random_range(0..remaining_indices.len())];
                selected.set_row(parent_index, &population.row(idx));
                parent_index += 1;
                remaining_spots -= 1;

                // Remove selected index
                if let Some(pos) = remaining_indices.iter().position(|&x| x == idx) {
                    remaining_indices.swap_remove(pos);
                }
            } else {
                let r = self.rng.random::<f64>();
                let mut cumsum = T::zero();

                for &idx in &remaining_indices {
                    cumsum += residual_probabilities[idx] / total_residual;
                    if T::from_f64(r).unwrap() <= cumsum {
                        selected.set_row(parent_index, &population.row(idx));
                        parent_index += 1;
                        remaining_spots -= 1;

                        // Remove selected index and reset its probability
                        if let Some(pos) = remaining_indices.iter().position(|&x| x == idx) {
                            remaining_indices.swap_remove(pos);
                        }
                        residual_probabilities[idx] = T::zero();
                        break;
                    }
                }
            }
        }

        selected
    }
}

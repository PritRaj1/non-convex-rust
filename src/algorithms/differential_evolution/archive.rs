use crate::utils::opt_prob::FloatNumber as FloatNum;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rand::Rng;

pub struct Archive<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub solutions: Vec<OVector<T, D>>,
    pub fitness: Vec<T>,
    pub max_size: usize,
}

impl<T, D> Archive<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub fn new(max_size: usize) -> Self {
        Self {
            solutions: Vec::with_capacity(max_size),
            fitness: Vec::with_capacity(max_size),
            max_size,
        }
    }

    pub fn add_solution(&mut self, solution: OVector<T, D>, fitness: T) {
        if self.solutions.len() < self.max_size {
            self.solutions.push(solution);
            self.fitness.push(fitness);
        } else if let Some(worst_idx) = self
            .fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
        {
            if fitness > self.fitness[worst_idx] {
                self.solutions[worst_idx] = solution;
                self.fitness[worst_idx] = fitness;
            }
        }
    }

    pub fn get_random_solution(&self) -> Option<&OVector<T, D>> {
        if self.solutions.is_empty() {
            None
        } else {
            let idx = rand::rng().random_range(0..self.solutions.len());
            Some(&self.solutions[idx])
        }
    }

    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    pub fn len(&self) -> usize {
        self.solutions.len()
    }
}

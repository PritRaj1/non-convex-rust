use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::utils::opt_prob::FloatNumber as FloatNum;

pub trait MutationOperator<T: FloatNum, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    fn mutate(
        &self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        generation: usize,
    ) -> OVector<T, D>;
}

pub struct Gaussian {
    pub mutation_rate: f64,
    pub sigma: f64,
}

impl Gaussian {
    pub fn new(mutation_rate: f64, sigma: f64) -> Self {
        Self {
            mutation_rate,
            sigma,
        }
    }
}

impl<T, D> MutationOperator<T, D> for Gaussian
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    fn mutate(
        &self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        _generation: usize,
    ) -> OVector<T, D> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, self.sigma).unwrap();
        let mut mutated = individual.clone();

        for i in 0..individual.len() {
            if rng.random::<f64>() < self.mutation_rate {
                let noise = T::from_f64(normal.sample(&mut rng)).unwrap();
                mutated[i] = (mutated[i] + noise).clamp(bounds.0, bounds.1);
            }
        }
        mutated
    }
}

pub struct Uniform {
    pub mutation_rate: f64,
}

impl Uniform {
    pub fn new(mutation_rate: f64) -> Self {
        Self { mutation_rate }
    }
}

impl<T, D> MutationOperator<T, D> for Uniform
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    fn mutate(
        &self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        _generation: usize,
    ) -> OVector<T, D> {
        let mut rng = rand::rng();
        let mut mutated = individual.clone();

        for i in 0..individual.len() {
            if rng.random::<f64>() < self.mutation_rate {
                mutated[i] = T::from_f64(
                    rng.random_range(bounds.0.to_f64().unwrap()..bounds.1.to_f64().unwrap()),
                )
                .unwrap();
            }
        }
        mutated
    }
}

pub struct NonUniform {
    pub mutation_rate: f64,
    pub b: f64, // Shape parameter
    pub max_generations: usize,
}

impl NonUniform {
    pub fn new(mutation_rate: f64, b: f64, max_generations: usize) -> Self {
        Self {
            mutation_rate,
            b,
            max_generations,
        }
    }
}

impl<T, D> MutationOperator<T, D> for NonUniform
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    fn mutate(
        &self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        generation: usize,
    ) -> OVector<T, D> {
        let mut rng = rand::rng();
        let mut mutated = individual.clone();
        let r = T::from_f64(rng.random::<f64>() * generation as f64 / self.max_generations as f64)
            .unwrap();

        for i in 0..individual.len() {
            if rng.random::<f64>() < self.mutation_rate {
                let delta = if rng.random_bool(0.5) {
                    bounds.1 - mutated[i]
                } else {
                    mutated[i] - bounds.0
                };

                let power = T::from_f64(
                    (T::one() - r).to_f64().unwrap().powf(self.b) * rng.random::<f64>(),
                )
                .unwrap();

                if rng.random_bool(0.5) {
                    mutated[i] += delta * power;
                } else {
                    mutated[i] -= delta * power;
                }

                mutated[i] = mutated[i].clamp(bounds.0, bounds.1);
            }
        }
        mutated
    }
}

pub struct Polynomial {
    pub mutation_rate: f64,
    pub eta_m: f64, // Distribution index
}

impl Polynomial {
    pub fn new(mutation_rate: f64, eta_m: f64) -> Self {
        Self {
            mutation_rate,
            eta_m,
        }
    }
}

impl<T, D> MutationOperator<T, D> for Polynomial
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    fn mutate(
        &self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        _generation: usize,
    ) -> OVector<T, D> {
        let mut rng = rand::rng();
        let mut mutated = individual.clone();

        for i in 0..individual.len() {
            if rng.random::<f64>() < self.mutation_rate {
                let r = rng.random::<f64>();
                let delta = if r < 0.5 {
                    (2.0 * r).powf(1.0 / (self.eta_m + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - r)).powf(1.0 / (self.eta_m + 1.0))
                };

                mutated[i] = (mutated[i] + T::from_f64(delta).unwrap() * (bounds.1 - bounds.0))
                    .clamp(bounds.0, bounds.1);
            }
        }
        mutated
    }
}

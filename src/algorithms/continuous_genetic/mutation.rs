use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::marker::PhantomData;

use crate::utils::opt_prob::FloatNumber as FloatNum;

pub trait MutationOperator<T: FloatNum, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    fn mutate(
        &mut self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        generation: usize,
    ) -> OVector<T, D>;
}

#[derive(Clone)]
pub enum MutationOperatorEnum<T: FloatNum, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    Gaussian(Gaussian, PhantomData<(T, D)>),
    Uniform(Uniform, PhantomData<(T, D)>),
    NonUniform(NonUniform, PhantomData<(T, D)>),
    Polynomial(Polynomial, PhantomData<(T, D)>),
}

impl<T: FloatNum, D: Dim> MutationOperator<T, D> for MutationOperatorEnum<T, D>
where
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    fn mutate(
        &mut self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        generation: usize,
    ) -> OVector<T, D> {
        match self {
            MutationOperatorEnum::Gaussian(op, _) => op.mutate(individual, bounds, generation),
            MutationOperatorEnum::Uniform(op, _) => op.mutate(individual, bounds, generation),
            MutationOperatorEnum::NonUniform(op, _) => op.mutate(individual, bounds, generation),
            MutationOperatorEnum::Polynomial(op, _) => op.mutate(individual, bounds, generation),
        }
    }
}

#[derive(Clone)]
pub struct Gaussian {
    pub mutation_rate: f64,
    pub sigma: f64,
    rng: StdRng,
}

impl Gaussian {
    pub fn new(mutation_rate: f64, sigma: f64, seed: u64) -> Self {
        Self {
            mutation_rate,
            sigma,
            rng: StdRng::seed_from_u64(seed),
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
        &mut self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        _generation: usize,
    ) -> OVector<T, D> {
        let normal = Normal::new(0.0, self.sigma).unwrap();
        let mut mutated = individual.clone();

        for i in 0..individual.len() {
            if self.rng.random::<f64>() < self.mutation_rate {
                let noise = T::from_f64(normal.sample(&mut self.rng)).unwrap();
                mutated[i] = (mutated[i] + noise).clamp(bounds.0, bounds.1);
            }
        }
        mutated
    }
}

#[derive(Clone)]
pub struct Uniform {
    pub mutation_rate: f64,
    rng: StdRng,
}

impl Uniform {
    pub fn new(mutation_rate: f64, seed: u64) -> Self {
        Self {
            mutation_rate,
            rng: StdRng::seed_from_u64(seed),
        }
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
        &mut self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        _generation: usize,
    ) -> OVector<T, D> {
        let mut mutated = individual.clone();

        for i in 0..individual.len() {
            if self.rng.random::<f64>() < self.mutation_rate {
                mutated[i] = T::from_f64(
                    self.rng
                        .random_range(bounds.0.to_f64().unwrap()..bounds.1.to_f64().unwrap()),
                )
                .unwrap();
            }
        }
        mutated
    }
}

#[derive(Clone)]
pub struct NonUniform {
    pub mutation_rate: f64,
    pub b: f64, // Shape parameter
    pub max_generations: usize,
    rng: StdRng,
}

impl NonUniform {
    pub fn new(mutation_rate: f64, b: f64, max_generations: usize, seed: u64) -> Self {
        Self {
            mutation_rate,
            b,
            max_generations,
            rng: StdRng::seed_from_u64(seed),
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
        &mut self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        generation: usize,
    ) -> OVector<T, D> {
        let mut mutated = individual.clone();
        let r =
            T::from_f64(self.rng.random::<f64>() * generation as f64 / self.max_generations as f64)
                .unwrap();

        for i in 0..individual.len() {
            if self.rng.random::<f64>() < self.mutation_rate {
                let delta = if self.rng.random_bool(0.5) {
                    bounds.1 - mutated[i]
                } else {
                    mutated[i] - bounds.0
                };

                let power = T::from_f64(
                    (T::one() - r).to_f64().unwrap().powf(self.b) * self.rng.random::<f64>(),
                )
                .unwrap();

                if self.rng.random_bool(0.5) {
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

#[derive(Clone)]
pub struct Polynomial {
    pub mutation_rate: f64,
    pub eta_m: f64, // Distribution index
    rng: StdRng,
}

impl Polynomial {
    pub fn new(mutation_rate: f64, eta_m: f64, seed: u64) -> Self {
        Self {
            mutation_rate,
            eta_m,
            rng: StdRng::seed_from_u64(seed),
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
        &mut self,
        individual: &OVector<T, D>,
        bounds: (T, T),
        _generation: usize,
    ) -> OVector<T, D> {
        let mut mutated = individual.clone();

        for i in 0..individual.len() {
            if self.rng.random::<f64>() < self.mutation_rate {
                let r = self.rng.random::<f64>();
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

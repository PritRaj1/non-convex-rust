use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rand::Rng;
use rand_distr::{Cauchy, Distribution, Normal};
use std::marker::PhantomData;

use crate::utils::alg_conf::tabu_conf::NeighborhoodStrategy;
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub enum NeighborhoodGenerator<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    Uniform(UniformNeighborhood),
    Gaussian(GaussianNeighborhood),
    Cauchy(CauchyNeighborhood),
    Adaptive(AdaptiveNeighborhood),
    _Phantom(PhantomData<(T, D)>),
}

impl<T, D> NeighborhoodGenerator<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    pub fn generate_neighbor(&self, current: &OVector<T, D>, rng: &mut impl Rng) -> OVector<T, D> {
        match self {
            NeighborhoodGenerator::Uniform(gen) => gen.generate_neighbor(current, rng),
            NeighborhoodGenerator::Gaussian(gen) => gen.generate_neighbor(current, rng),
            NeighborhoodGenerator::Cauchy(gen) => gen.generate_neighbor(current, rng),
            NeighborhoodGenerator::Adaptive(gen) => gen.generate_neighbor(current, rng),
            NeighborhoodGenerator::_Phantom(_) => unreachable!(),
        }
    }

    pub fn update_parameters(&mut self, success_rate: f64, improvement_rate: f64) {
        match self {
            NeighborhoodGenerator::Uniform(gen) => {
                gen.update_parameters(success_rate, improvement_rate)
            }
            NeighborhoodGenerator::Gaussian(gen) => {
                gen.update_parameters(success_rate, improvement_rate)
            }
            NeighborhoodGenerator::Cauchy(gen) => {
                gen.update_parameters(success_rate, improvement_rate)
            }
            NeighborhoodGenerator::Adaptive(gen) => {
                gen.update_parameters(success_rate, improvement_rate)
            }
            NeighborhoodGenerator::_Phantom(_) => {}
        }
    }
}

pub struct UniformNeighborhood {
    step_size: f64,
    prob: f64,
}

impl UniformNeighborhood {
    pub fn new(step_size: f64, prob: f64) -> Self {
        Self { step_size, prob }
    }

    fn generate_neighbor<T, D>(&self, current: &OVector<T, D>, rng: &mut impl Rng) -> OVector<T, D>
    where
        T: FloatNum + Send + Sync,
        D: Dim,
        OVector<T, D>: Send + Sync,
        DefaultAllocator: Allocator<D>,
    {
        let mut neighbor = current.clone();
        neighbor.iter_mut().for_each(|val| {
            if rng.random_bool(self.prob) {
                *val += T::from_f64(rng.random_range(-self.step_size..self.step_size)).unwrap();
            }
        });
        neighbor
    }

    // No adaption in uniform
    fn update_parameters(&mut self, _success_rate: f64, _improvement_rate: f64) {}
}

pub struct GaussianNeighborhood {
    sigma: f64,
    prob: f64,
}

impl GaussianNeighborhood {
    pub fn new(sigma: f64, prob: f64) -> Self {
        Self { sigma, prob }
    }

    fn generate_neighbor<T, D>(&self, current: &OVector<T, D>, rng: &mut impl Rng) -> OVector<T, D>
    where
        T: FloatNum + Send + Sync,
        D: Dim,
        OVector<T, D>: Send + Sync,
        DefaultAllocator: Allocator<D>,
    {
        let mut neighbor = current.clone();
        let normal = Normal::new(0.0, self.sigma).unwrap();

        neighbor.iter_mut().for_each(|val| {
            if rng.random_bool(self.prob) {
                let perturbation = T::from_f64(normal.sample(rng)).unwrap();
                *val += perturbation;
            }
        });
        neighbor
    }

    // No adaption in Gaussian
    fn update_parameters(&mut self, _success_rate: f64, _improvement_rate: f64) {}
}

pub struct CauchyNeighborhood {
    scale: f64,
    prob: f64,
}

impl CauchyNeighborhood {
    pub fn new(scale: f64, prob: f64) -> Self {
        Self { scale, prob }
    }

    fn generate_neighbor<T, D>(&self, current: &OVector<T, D>, rng: &mut impl Rng) -> OVector<T, D>
    where
        T: FloatNum + Send + Sync,
        D: Dim,
        OVector<T, D>: Send + Sync,
        DefaultAllocator: Allocator<D>,
    {
        let mut neighbor = current.clone();
        let cauchy = Cauchy::new(0.0, self.scale).unwrap();

        neighbor.iter_mut().for_each(|val| {
            if rng.random_bool(self.prob) {
                let perturbation = T::from_f64(cauchy.sample(rng)).unwrap();
                *val += perturbation;
            }
        });
        neighbor
    }

    // No adaption in Cauchy
    fn update_parameters(&mut self, _success_rate: f64, _improvement_rate: f64) {}
}

pub struct AdaptiveNeighborhood {
    base_step: f64,
    adaptation_rate: f64,
    prob: f64,
    current_step: f64,
}

impl AdaptiveNeighborhood {
    pub fn new(base_step: f64, adaptation_rate: f64, prob: f64) -> Self {
        Self {
            base_step,
            adaptation_rate,
            prob,
            current_step: base_step,
        }
    }

    fn generate_neighbor<T, D>(&self, current: &OVector<T, D>, rng: &mut impl Rng) -> OVector<T, D>
    where
        T: FloatNum + Send + Sync,
        D: Dim,
        OVector<T, D>: Send + Sync,
        DefaultAllocator: Allocator<D>,
    {
        let mut neighbor = current.clone();
        neighbor.iter_mut().for_each(|val| {
            if rng.random_bool(self.prob) {
                *val +=
                    T::from_f64(rng.random_range(-self.current_step..self.current_step)).unwrap();
            }
        });
        neighbor
    }

    fn update_parameters(&mut self, success_rate: f64, improvement_rate: f64) {
        if success_rate < 0.2 {
            self.current_step *= 1.0 + self.adaptation_rate; // Low success - increase exploration
        } else if success_rate > 0.6 && improvement_rate > 1e-4 {
            self.current_step *= 1.0 - self.adaptation_rate * 0.3; // High success - fine-tune
        }

        // Keep step size within reasonable bounds
        self.current_step = self
            .current_step
            .clamp(self.base_step * 0.1, self.base_step * 10.0);
    }
}

pub fn create_neighborhood_generator<T, D>(
    strategy: &NeighborhoodStrategy,
) -> NeighborhoodGenerator<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    match strategy {
        NeighborhoodStrategy::Uniform { step_size, prob } => {
            NeighborhoodGenerator::Uniform(UniformNeighborhood::new(*step_size, *prob))
        }
        NeighborhoodStrategy::Gaussian { sigma, prob } => {
            NeighborhoodGenerator::Gaussian(GaussianNeighborhood::new(*sigma, *prob))
        }
        NeighborhoodStrategy::Cauchy { scale, prob } => {
            NeighborhoodGenerator::Cauchy(CauchyNeighborhood::new(*scale, *prob))
        }
        NeighborhoodStrategy::Adaptive {
            base_step,
            adaptation_rate,
        } => NeighborhoodGenerator::Adaptive(AdaptiveNeighborhood::new(
            *base_step,
            *adaptation_rate,
            0.3,
        )),
    }
}

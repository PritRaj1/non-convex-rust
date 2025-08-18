use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rand::Rng;

use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

pub enum AcceptanceType {
    Metropolis,
    MALA,
}

pub trait AcceptanceCriterion<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn accept(
        &self,
        current_x: &OVector<T, D>,
        current_fitness: T,
        new_x: &OVector<T, D>,
        new_fitness: T,
        temperature: T,
        step_size: T,
    ) -> bool;
}

pub struct MetropolisAcceptance<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub acceptance_type: AcceptanceType,
    pub prob: OptProb<T, D>,
    k: T, // Boltzmann constant
}

impl<T, D> MetropolisAcceptance<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub fn new(prob: OptProb<T, D>, generic_x: OVector<T, D>) -> Self {
        let acceptance_type = if prob.objective.gradient(&generic_x).is_some() {
            AcceptanceType::MALA
        } else {
            AcceptanceType::Metropolis
        };

        Self {
            acceptance_type,
            prob,
            k: T::from_f64(1.0).unwrap(),
        }
    }

    pub fn accept(
        &self,
        current_x: &OVector<T, D>,
        current_fitness: T,
        new_x: &OVector<T, D>,
        new_fitness: T,
        temperature: T,
        step_size: T,
    ) -> bool {
        let mut rng = rand::rng();

        if new_fitness > current_fitness {
            return true;
        }

        let delta_f = new_fitness - current_fitness;

        let r = match self.acceptance_type {
            AcceptanceType::Metropolis => (delta_f / (temperature * self.k)).exp(),
            AcceptanceType::MALA => {
                let grad = self.prob.objective.gradient(current_x).unwrap();
                let proposal_grad = self.prob.objective.gradient(new_x).unwrap();

                let langevin_correction =
                    -((new_x - current_x - grad.clone() * step_size * temperature)
                        .dot(&(new_x - current_x - grad * step_size * temperature))
                        / (T::from_f64(4.0).unwrap() * step_size * temperature))
                        + ((current_x - new_x - proposal_grad.clone() * step_size * temperature)
                            .dot(&(current_x - new_x - proposal_grad * step_size * temperature))
                            / (T::from_f64(4.0).unwrap() * step_size * temperature));

                (delta_f / (self.k * temperature) + langevin_correction).exp()
            }
        };

        rng.random::<f64>() < r.to_f64().unwrap()
    }
}

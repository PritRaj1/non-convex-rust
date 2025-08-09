use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use rand::Rng;
use rand_distr::{Normal, StandardNormal};
use rayon::prelude::*;

use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

pub enum MoveType {
    RandomDrift,
    MALA,
}

pub struct GaussianGenerator<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub move_type: MoveType,
    pub prob: OptProb<T, D>,
    pub mala_step_size: T,
}

impl<T, D> GaussianGenerator<T, D>
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    pub fn new(prob: OptProb<T, D>, generic_x: OVector<T, D>, mala_step_size: T) -> Self {
        let move_type = if prob.objective.gradient(&generic_x).is_some() {
            MoveType::MALA
        } else {
            MoveType::RandomDrift
        };

        Self {
            move_type,
            prob,
            mala_step_size,
        }
    }

    pub fn generate(
        &self,
        current: &OVector<T, D>,
        step_size: f64,
        bounds: (T, T),
        temperature: T,
    ) -> OVector<T, D> {
        match self.move_type {
            MoveType::RandomDrift => self.random_drift(current, step_size, bounds),
            MoveType::MALA => self.mala_move(current, temperature, bounds),
        }
    }

    fn random_drift(
        &self,
        current: &OVector<T, D>,
        step_size: f64,
        bounds: (T, T),
    ) -> OVector<T, D> {
        let mut neighbor = current.clone();

        neighbor
            .as_mut_slice()
            .par_chunks_mut(1)
            .enumerate()
            .for_each(|(i, val)| {
                let mut rng = rand::rng();
                let step = T::from_f64(rng.sample::<f64, _>(Normal::new(0.0, step_size).unwrap()))
                    .unwrap();
                val[0] = (current[i] + step).clamp(bounds.0, bounds.1);
            });

        neighbor
    }

    fn mala_move(&self, current: &OVector<T, D>, temperature: T, bounds: (T, T)) -> OVector<T, D> {
        let mut rng = rand::rng();
        let step = self.mala_step_size * temperature;

        let grad = self.prob.objective.gradient(current).unwrap();
        let drift = grad * step;

        let noise = OVector::<T, D>::from_fn_generic(D::from_usize(current.len()), U1, |_, _| {
            T::from_f64(rng.sample::<f64, _>(StandardNormal)).unwrap()
        }) * (step * T::from_f64(2.0).unwrap()).sqrt();

        let mut new_pos = current + drift + noise;
        for i in 0..new_pos.len() {
            new_pos[i] = new_pos[i].clamp(bounds.0, bounds.1); // Not valid without Jacobian, but hey!
        }

        new_pos
    }
}

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Normal, StandardNormal};
use rayon::prelude::*;

use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

#[derive(Clone)]
pub enum MoveType {
    RandomDrift,
    MALA,
}

#[derive(Clone)]
pub struct GaussianGenerator<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub move_type: MoveType,
    pub prob: OptProb<T, D>,
    pub mala_step_size: T,
    rng: StdRng,
    seed: u64,
}

impl<T, D> GaussianGenerator<T, D>
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    pub fn new(
        prob: OptProb<T, D>,
        generic_x: OVector<T, D>,
        mala_step_size: T,
        seed: u64,
    ) -> Self {
        let move_type = if prob.objective.gradient(&generic_x).is_some() {
            MoveType::MALA
        } else {
            MoveType::RandomDrift
        };

        Self {
            move_type,
            prob,
            mala_step_size,
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    pub fn generate(
        &mut self,
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
        &mut self,
        current: &OVector<T, D>,
        step_size: f64,
        bounds: (T, T),
    ) -> OVector<T, D> {
        let mut neighbor = current.clone();

        let steps: Vec<T> = (0..current.len())
            .into_par_iter()
            .map_init(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    // Note: iteration would need to be passed in, using a counter for now
                    StdRng::seed_from_u64(self.seed + thread_id as u64)
                },
                |rng, _| T::from_f64(rng.sample(Normal::new(0.0, step_size).unwrap())).unwrap(),
            )
            .collect();

        for (i, step) in steps.into_iter().enumerate() {
            let new_pos = current[i] + step;

            // Reflective bounds
            if new_pos < bounds.0 {
                let reflected_pos = bounds.0 + (bounds.0 - new_pos);
                neighbor[i] = reflected_pos.clamp(bounds.0, bounds.1);
            } else if new_pos > bounds.1 {
                let reflected_pos = bounds.1 - (new_pos - bounds.1);
                neighbor[i] = reflected_pos.clamp(bounds.0, bounds.1);
            } else {
                neighbor[i] = new_pos;
            }
        }

        neighbor
    }

    fn mala_move(
        &mut self,
        current: &OVector<T, D>,
        temperature: T,
        bounds: (T, T),
    ) -> OVector<T, D> {
        let step = self.mala_step_size * temperature;

        let grad = self.prob.objective.gradient(current).unwrap();
        let drift = grad * step;

        let noise = OVector::<T, D>::from_fn_generic(D::from_usize(current.len()), U1, |_, _| {
            T::from_f64(self.rng.sample::<f64, _>(StandardNormal)).unwrap()
        }) * (step * T::from_f64(2.0).unwrap()).sqrt();

        let mut new_pos = current + drift + noise;

        // Reflective bounds (does not preserve detailed balance)
        for i in 0..new_pos.len() {
            if new_pos[i] < bounds.0 {
                let reflected_pos = bounds.0 + (bounds.0 - new_pos[i]);
                new_pos[i] = reflected_pos.clamp(bounds.0, bounds.1);
            } else if new_pos[i] > bounds.1 {
                let reflected_pos = bounds.1 - (new_pos[i] - bounds.1);
                new_pos[i] = reflected_pos.clamp(bounds.0, bounds.1);
            }
        }

        new_pos
    }
}

use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use rand::{rngs::StdRng, Rng};

pub struct Particle<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    pub position: OVector<T, D>,
    pub velocity: OVector<T, D>,
    pub best_position: OVector<T, D>,
    pub best_fitness: T,
    pub improvement_counter: usize,
    pub stagnation_counter: usize,
    rng: StdRng,
}

impl<T, D> Particle<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    pub fn new(position: OVector<T, D>, velocity: OVector<T, D>, fitness: T, rng: StdRng) -> Self {
        Self {
            position: position.clone(),
            velocity,
            best_position: position,
            best_fitness: fitness,
            improvement_counter: 0,
            stagnation_counter: 0,
            rng,
        }
    }

    pub fn update_velocity_and_position(
        &mut self,
        global_best: &OVector<T, D>,
        w: T,
        c1: T,
        c2: T,
        opt_prob: &OptProb<T, D>,
        bounds: (T, T),
    ) {
        for i in 0..self.velocity.len() {
            let r1 = T::from_f64(self.rng.random::<f64>()).unwrap();
            let r2 = T::from_f64(self.rng.random::<f64>()).unwrap();

            let cognitive = c1 * r1 * (self.best_position[i] - self.position[i]);
            let social = c2 * r2 * (global_best[i] - self.position[i]);

            // Clamp based on search space size
            let search_space_size = bounds.1 - bounds.0;
            let v_max = search_space_size * T::from_f64(0.2).unwrap(); // More aggressive clamping

            self.velocity[i] = (w * self.velocity[i] + cognitive + social).clamp(-v_max, v_max);
        }

        // Reflective boundary
        let new_positions: Vec<T> = self
            .position
            .iter()
            .zip(self.velocity.iter())
            .map(|(&p, &v)| {
                let new_pos = p + v;
                if new_pos < bounds.0 {
                    let reflected_pos = bounds.0 + (bounds.0 - new_pos);
                    reflected_pos.clamp(bounds.0, bounds.1)
                } else if new_pos > bounds.1 {
                    let reflected_pos = bounds.1 - (new_pos - bounds.1);
                    reflected_pos.clamp(bounds.0, bounds.1)
                } else {
                    new_pos
                }
            })
            .collect();

        let final_position = OVector::<T, D>::from_vec_generic(
            D::from_usize(new_positions.len()),
            U1,
            new_positions,
        );
        self.position = final_position;

        // Only update best position if new position is better AND feasible
        let new_fitness = opt_prob.evaluate(&self.position);
        if new_fitness > self.best_fitness && opt_prob.is_feasible(&self.position) {
            self.best_fitness = new_fitness;
            self.best_position = self.position.clone();
            self.improvement_counter += 1;
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }
    }

    pub fn is_stagnated(&self, threshold: usize) -> bool {
        self.stagnation_counter > threshold
    }

    // Iterations since last improvement
    pub fn age(&self) -> usize {
        self.stagnation_counter
    }

    pub fn reset_stagnation(&mut self) {
        self.stagnation_counter = 0;
    }
}

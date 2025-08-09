use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use rand::Rng;

pub struct Particle<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    pub position: OVector<T, D>,
    pub velocity: OVector<T, D>,
    pub best_position: OVector<T, D>,
    pub best_fitness: T,
}

impl<T: FloatNum, D: Dim> Particle<T, D>
where
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    pub fn new(position: OVector<T, D>, velocity: OVector<T, D>, fitness: T) -> Self {
        Self {
            position: position.clone(),
            velocity,
            best_position: position,
            best_fitness: fitness,
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
        let mut rng = rand::rng();

        // Update velocity
        for i in 0..self.velocity.len() {
            let r1 = T::from_f64(rng.random::<f64>()).unwrap();
            let r2 = T::from_f64(rng.random::<f64>()).unwrap();

            let cognitive = c1 * r1 * (self.best_position[i] - self.position[i]);
            let social = c2 * r2 * (global_best[i] - self.position[i]);

            // Add velocity clamping
            let v_max = (bounds.1 - bounds.0) * T::from_f64(0.1).unwrap();
            self.velocity[i] = (w * self.velocity[i] + cognitive + social).clamp(-v_max, v_max);
        }

        // Update position with bounds checking only
        let new_positions: Vec<T> = self
            .position
            .iter()
            .zip(self.velocity.iter())
            .map(|(&p, &v)| {
                let new_pos = p + v;
                new_pos.clamp(bounds.0, bounds.1)
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
        }
    }
}

use crate::algorithms::multi_swarm::swarm::Swarm;
use crate::utils::config::MSPOConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, U1};

pub struct InformationExchange<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<U1, D>,
{
    conf: MSPOConf,
    opt_prob: OptProb<T, D>,
}

impl<T, D> InformationExchange<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<U1, D>,
{
    pub fn new(conf: MSPOConf, opt_prob: OptProb<T, D>) -> Self {
        Self { conf, opt_prob }
    }

    pub fn exchange_information(&self, swarms: &mut [Swarm<T, D>], stagnation_counter: usize) {
        let exchange_ratio = self.compute_adaptive_exchange_ratio(stagnation_counter);
        let num_particles_to_exchange =
            (self.conf.swarm_size as f64 * exchange_ratio).round() as usize;

        if num_particles_to_exchange == 0 {
            return;
        }

        for i in 0..swarms.len() {
            let source_idx = i;
            let target_idx = (i + 1) % swarms.len();

            // Use split_at_mut to get mutable references to both swarms
            if source_idx < target_idx {
                let (left, right) = swarms.split_at_mut(target_idx);
                let source_swarm = &mut left[source_idx];
                let target_swarm = &mut right[0];
                self.exchange_between_swarms(source_swarm, target_swarm, num_particles_to_exchange);
            } else if source_idx > target_idx {
                let (left, right) = swarms.split_at_mut(source_idx);
                let source_swarm = &mut right[0];
                let target_swarm = &mut left[target_idx];
                self.exchange_between_swarms(source_swarm, target_swarm, num_particles_to_exchange);
            }
        }
    }

    fn exchange_between_swarms(
        &self,
        source_swarm: &mut Swarm<T, D>,
        target_swarm: &mut Swarm<T, D>,
        num_particles_to_exchange: usize,
    ) {
        // Find best particles from source swarm
        let mut best_particles: Vec<_> = source_swarm
            .particles
            .iter()
            .enumerate()
            .map(|(j, p)| (j, p.best_fitness))
            .collect();
        best_particles.sort_by(|(_, f1), (_, f2)| f2.partial_cmp(f1).unwrap());

        // Exchange best particles with target swarm's worst particles
        for (k, (source_particle_idx, _)) in best_particles
            .iter()
            .enumerate()
            .take(num_particles_to_exchange.min(best_particles.len()))
        {
            let target_particle_idx = k;

            if *source_particle_idx < source_swarm.particles.len()
                && target_particle_idx < target_swarm.particles.len()
            {
                // Only exchange if the source particle is significantly better
                let source_fitness = source_swarm.particles[*source_particle_idx].best_fitness;
                let target_fitness = target_swarm.particles[target_particle_idx].best_fitness;
                let better_fitness = source_fitness.max(target_fitness);
                let better_pos = if source_fitness > target_fitness {
                    source_swarm.particles[*source_particle_idx]
                        .best_position
                        .clone()
                } else {
                    target_swarm.particles[target_particle_idx]
                        .best_position
                        .clone()
                };

                if better_fitness
                    > target_swarm.particles[target_particle_idx].best_fitness
                        * (T::one() + T::from_f64(self.conf.improvement_threshold).unwrap())
                    && self.opt_prob.is_feasible(&better_pos)
                {
                    target_swarm.particles[target_particle_idx].best_position = better_pos;
                    target_swarm.particles[target_particle_idx].best_fitness = better_fitness;
                    break; // Only use the first better solution
                }
            }
        }
    }

    fn compute_adaptive_exchange_ratio(&self, stagnation_counter: usize) -> f64 {
        let base_ratio = self.conf.exchange_ratio;
        if stagnation_counter > 15 {
            base_ratio * 2.0 // More aggressive exchange when highly stagnated
        } else if stagnation_counter > 10 {
            base_ratio * 1.5 // Moderate increase
        } else {
            base_ratio
        }
    }
}

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1};
use rand::Rng;
use rayon::prelude::*;

use crate::utils::config::MSPOConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

use crate::algorithms::multi_swarm::particle::Particle;

#[derive(Clone)]
pub struct SwarmConfig<'a, T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<Dyn, D>,
{
    pub num_particles: usize,
    pub dim: usize,
    pub w: T,
    pub c1: T,
    pub c2: T,
    pub bounds: (T, T),
    pub opt_prob: &'a OptProb<T, D>,
    pub init_pop: OMatrix<T, Dyn, D>,
}

pub struct Swarm<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D>,
{
    pub particles: Vec<Particle<T, D>>,
    pub global_best_position: OVector<T, D>,
    pub global_best_fitness: T,
    pub w: T,
    pub c1: T,
    pub c2: T,
    pub x_min: f64,
    pub x_max: f64,
}

impl<T, D> Swarm<T, D>
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<Dyn, D>,
{
    pub fn new(config: SwarmConfig<T, D>) -> Self {
        let particles: Vec<_> = (0..config.num_particles)
            .into_par_iter()
            .map(|i| {
                let mut rng = rand::rng();
                let mut position = OVector::<T, D>::zeros_generic(D::from_usize(config.dim), U1);
                let fitness;

                if i < config.init_pop.nrows() {
                    // Use initial population if available
                    position = config.init_pop.row(i).transpose();
                    fitness = config.opt_prob.evaluate(&position);
                } else {
                    // Generate random position if needed
                    loop {
                        let values = (0..config.dim).map(|_| {
                            let r = T::from_f64(rng.random::<f64>()).unwrap();
                            config.bounds.0 + (config.bounds.1 - config.bounds.0) * r
                        });
                        let position: OVector<T, D> =
                            OVector::from_iterator_generic(D::from_usize(config.dim), U1, values);

                        if config.opt_prob.is_feasible(&position) {
                            fitness = config.opt_prob.evaluate(&position);
                            break;
                        }
                    }
                }

                let values = (0..config.dim).map(|_| {
                    let r = T::from_f64(rng.random::<f64>()).unwrap();
                    (config.bounds.1 - config.bounds.0)
                        * (r - T::from_f64(0.5).unwrap())
                        * T::from_f64(0.1).unwrap()
                });

                let velocity: OVector<T, D> =
                    OVector::from_iterator_generic(D::from_usize(config.dim), U1, values);

                Particle::new(position, velocity, fitness)
            })
            .collect();

        let mut best_fitness = T::neg_infinity();
        let mut best_position = OVector::<T, D>::zeros_generic(D::from_usize(config.dim), U1);

        for particle in &particles {
            if particle.best_fitness > best_fitness {
                best_fitness = particle.best_fitness;
                best_position = particle.position.clone();
            }
        }

        Self {
            particles,
            global_best_position: best_position,
            global_best_fitness: best_fitness,
            w: config.w,
            c1: config.c1,
            c2: config.c2,
            x_min: config.bounds.0.to_f64().unwrap(),
            x_max: config.bounds.1.to_f64().unwrap(),
        }
    }

    pub fn update(&mut self, opt_prob: &OptProb<T, D>) {
        let bounds = (
            T::from_f64(self.x_min).unwrap(),
            T::from_f64(self.x_max).unwrap(),
        );

        self.particles.par_iter_mut().for_each(|particle| {
            particle.update_velocity_and_position(
                &self.global_best_position,
                self.w,
                self.c1,
                self.c2,
                opt_prob,
                bounds,
            );
        });

        let best_particle = self
            .particles
            .par_iter()
            .reduce_with(|p1, p2| {
                if p1.best_fitness > p2.best_fitness {
                    p1
                } else {
                    p2
                }
            })
            .unwrap();

        if best_particle.best_fitness > self.global_best_fitness {
            self.global_best_fitness = best_particle.best_fitness;
            self.global_best_position = best_particle.best_position.clone();
        }
    }
}

pub fn initialize_swarms<T, N, D>(
    conf: &MSPOConf,
    dim: usize,
    init_pop: &OMatrix<T, N, D>,
    opt_prob: &OptProb<T, D>,
) -> Vec<Swarm<T, D>>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<D, D>,
{
    let particles_per_swarm = conf.swarm_size;
    let pop_per_swarm = init_pop.nrows() / conf.num_swarms;

    // Find several promising regions
    let mut promising_centers: Vec<OVector<T, D>> = Vec::new();
    let mut sorted_indices: Vec<usize> = (0..init_pop.nrows()).collect();
    sorted_indices.sort_by(|&i, &j| {
        let fi = opt_prob.evaluate(&init_pop.row(i).transpose());
        let fj = opt_prob.evaluate(&init_pop.row(j).transpose());
        fj.partial_cmp(&fi).unwrap()
    });

    // Select diverse centers from top solutions
    for &idx in sorted_indices.iter().take(conf.num_swarms) {
        let center = init_pop.row(idx).transpose();
        if promising_centers.iter().all(|c| {
            // Ensure centers are sufficiently far apart
            let dist = (c - &center).dot(&(c - &center)).sqrt();
            dist > T::from_f64(0.1 * (conf.x_max - conf.x_min)).unwrap()
        }) {
            promising_centers.push(center);
        }
    }

    // Initialize swarms around these promising regions
    (0..conf.num_swarms)
        .into_par_iter()
        .map(|i| {
            let center = if i < promising_centers.len() {
                promising_centers[i].clone()
            } else {
                // Random center for remaining swarms
                OVector::<T, D>::from_iterator_generic(
                    D::from_usize(dim),
                    U1,
                    (0..dim).map(|_| {
                        T::from_f64(conf.x_min + rand::random::<f64>() * (conf.x_max - conf.x_min))
                            .unwrap()
                    }),
                )
            };

            // Initialize particles around the center
            let radius = T::from_f64(0.2 * (conf.x_max - conf.x_min)).unwrap(); // Local search radius
            let start_idx = i * pop_per_swarm;
            let mut swarm_pop: OMatrix<T, Dyn, D> =
                init_pop.rows(start_idx, particles_per_swarm).into_owned();

            // Adjust some particles to be near the center
            for j in 0..particles_per_swarm / 2 {
                for k in 0..dim {
                    let r = T::from_f64(rand::random::<f64>()).unwrap();
                    swarm_pop[(j, k)] = center[k] + (r - T::from_f64(0.5).unwrap()) * radius;
                }
            }

            Swarm::new(SwarmConfig {
                num_particles: particles_per_swarm,
                dim,
                w: T::from_f64(conf.w).unwrap(),
                c1: T::from_f64(conf.c1).unwrap(),
                c2: T::from_f64(conf.c2).unwrap(),
                bounds: (
                    T::from_f64(conf.x_min).unwrap(),
                    T::from_f64(conf.x_max).unwrap(),
                ),
                opt_prob,
                init_pop: swarm_pop,
            })
        })
        .collect()
}

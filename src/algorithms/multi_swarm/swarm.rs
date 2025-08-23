use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1};
use rand::{rngs::StdRng, Rng, SeedableRng};
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
    pub c1: T,
    pub c2: T,
    pub bounds: (T, T),
    pub opt_prob: &'a OptProb<T, D>,
    pub init_pop: OMatrix<T, Dyn, D>,
    pub inertia_start: T,
    pub inertia_end: T,
    pub max_iterations: usize,
}

pub struct Swarm<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<U1, D>,
{
    pub particles: Vec<Particle<T, D>>,
    pub global_best_position: OVector<T, D>,
    pub global_best_fitness: T,
    pub c1: T,
    pub c2: T,
    pub x_min: f64,
    pub x_max: f64,
    pub iteration_count: usize,
    pub improvement_history: Vec<T>,
    pub diversity_history: Vec<f64>,
    pub inertia_start: T,
    pub inertia_end: T,
    pub max_iterations: usize,
}

impl<T, D> Swarm<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<Dyn, D>,
{
    pub fn new(config: SwarmConfig<T, D>, seed: u64) -> Self {
        let base_rng = StdRng::seed_from_u64(seed);
        let particles: Vec<_> = (0..config.num_particles)
            .into_par_iter()
            .map_init(
                || base_rng.clone(),
                |rng, i| {
                    let mut position =
                        OVector::<T, D>::zeros_generic(D::from_usize(config.dim), U1);
                    let fitness;

                    if i < config.init_pop.nrows() {
                        position = config.init_pop.row(i).transpose();
                        fitness = config.opt_prob.evaluate(&position);
                    } else {
                        loop {
                            let values = (0..config.dim).map(|_| {
                                let r = T::from_f64(rng.random::<f64>()).unwrap();
                                config.bounds.0 + (config.bounds.1 - config.bounds.0) * r
                            });
                            let position: OVector<T, D> = OVector::from_iterator_generic(
                                D::from_usize(config.dim),
                                U1,
                                values,
                            );

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

                    Particle::new(position, velocity, fitness, rng.clone())
                },
            )
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
            c1: config.c1,
            c2: config.c2,
            x_min: config.bounds.0.to_f64().unwrap(),
            x_max: config.bounds.1.to_f64().unwrap(),
            iteration_count: 0,
            improvement_history: Vec::new(),
            diversity_history: Vec::new(),
            inertia_start: config.inertia_start,
            inertia_end: config.inertia_end,
            max_iterations: config.max_iterations,
        }
    }

    pub fn update(&mut self, opt_prob: &OptProb<T, D>) {
        let bounds = (
            T::from_f64(self.x_min).unwrap(),
            T::from_f64(self.x_max).unwrap(),
        );

        // Adaptive weight based on stagnation and diversity
        let adaptive_w = self.compute_adaptive_inertia();

        self.particles.par_iter_mut().for_each(|particle| {
            particle.update_velocity_and_position(
                &self.global_best_position,
                adaptive_w,
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
            let improvement = best_particle.best_fitness - self.global_best_fitness;
            self.improvement_history.push(improvement);

            self.global_best_fitness = best_particle.best_fitness;
            self.global_best_position = best_particle.best_position.clone();
        }

        let diversity = self.compute_diversity();
        self.diversity_history.push(diversity);

        self.iteration_count += 1;
    }

    // Linearly decrease from w_start to w_end over total run
    fn compute_adaptive_inertia(&self) -> T {
        let progress = (self.iteration_count as f64 / self.max_iterations as f64).min(1.0);

        self.inertia_start
            + (self.inertia_end - self.inertia_start) * T::from_f64(progress).unwrap()
    }

    // Avg distance between particles
    fn compute_diversity(&self) -> f64 {
        if self.particles.len() < 2 {
            return 0.0;
        }

        let pairs: Vec<_> = (0..self.particles.len())
            .flat_map(|i| (i + 1..self.particles.len()).map(move |j| (i, j)))
            .collect();

        if pairs.is_empty() {
            return 0.0;
        }

        let total_distance: f64 = pairs
            .par_iter()
            .map(|&(i, j)| {
                let distance = self
                    .euclidean_distance(&self.particles[i].position, &self.particles[j].position);
                distance.to_f64().unwrap()
            })
            .sum();

        total_distance / pairs.len() as f64
    }

    fn euclidean_distance(&self, v1: &OVector<T, D>, v2: &OVector<T, D>) -> T {
        let diff = v1 - v2;
        diff.dot(&diff).sqrt()
    }

    pub fn is_stagnated(&self, threshold: usize) -> bool {
        if self.improvement_history.len() < threshold {
            return false;
        }

        let recent_improvements: Vec<T> = self
            .improvement_history
            .iter()
            .rev()
            .take(threshold)
            .cloned()
            .collect();

        let max_improvement = recent_improvements
            .iter()
            .fold(T::neg_infinity(), |a, &b| a.max(b));
        max_improvement < T::epsilon()
    }

    pub fn average_improvement(&self, window_size: usize) -> T {
        if self.improvement_history.is_empty() {
            return T::zero();
        }

        let recent_improvements: Vec<T> = self
            .improvement_history
            .iter()
            .rev()
            .take(window_size.min(self.improvement_history.len()))
            .cloned()
            .collect();

        let len = recent_improvements.len();
        let sum = recent_improvements
            .into_iter()
            .fold(T::zero(), |acc, x| acc + x);
        sum / T::from_usize(len).unwrap()
    }

    pub fn current_diversity(&self) -> f64 {
        self.diversity_history.last().copied().unwrap_or(0.0)
    }
}

pub fn initialize_swarms<T, N, D>(
    conf: &MSPOConf,
    dim: usize,
    init_pop: &OMatrix<T, N, D>,
    opt_prob: &OptProb<T, D>,
    max_iter: usize,
    seed: u64,
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
    let mut promising_centers: Vec<OVector<T, D>> = Vec::new();

    let fitness_values: Vec<_> = (0..init_pop.nrows())
        .into_par_iter()
        .map(|i| {
            let individual = init_pop.row(i).transpose();
            (i, opt_prob.evaluate(&individual))
        })
        .collect();

    let mut sorted_indices: Vec<usize> = (0..init_pop.nrows()).collect();
    sorted_indices.sort_by(|&i, &j| {
        fitness_values[i]
            .1
            .partial_cmp(&fitness_values[j].1)
            .unwrap()
    });

    // Find diverse centers with min dist constraint
    for &idx in sorted_indices.iter().take(conf.num_swarms) {
        let center = init_pop.row(idx).transpose();

        // Centers should be far apart (for diversity)
        let min_distance = T::from_f64(0.3 * (conf.x_max - conf.x_min)).unwrap();

        if promising_centers.is_empty() {
            promising_centers.push(center);
        } else {
            let is_diverse = promising_centers.par_iter().all(|c| {
                let dist = (c - &center).dot(&(c - &center)).sqrt();
                dist > min_distance
            });

            if is_diverse {
                promising_centers.push(center);
            }
        }
    }

    let mut fill_rng = StdRng::seed_from_u64(seed + 1000);
    while promising_centers.len() < conf.num_swarms {
        let random_center = OVector::<T, D>::from_iterator_generic(
            D::from_usize(dim),
            U1,
            (0..dim).map(|_| {
                T::from_f64(conf.x_min + fill_rng.random::<f64>() * (conf.x_max - conf.x_min))
                    .unwrap()
            }),
        );
        promising_centers.push(random_center);
    }

    // Fill with promising, fallback to random
    let base_rng = StdRng::seed_from_u64(seed + 2000);
    (0..conf.num_swarms)
        .into_par_iter()
        .map_init(
            || base_rng.clone(),
            |rng, i| {
                let center = if i < promising_centers.len() {
                    promising_centers[i].clone()
                } else {
                    OVector::<T, D>::from_iterator_generic(
                        D::from_usize(dim),
                        U1,
                        (0..dim).map(|_| {
                            T::from_f64(
                                conf.x_min + rng.random::<f64>() * (conf.x_max - conf.x_min),
                            )
                            .unwrap()
                        }),
                    )
                };

                // Spawn particles around center with controlled spread
                let radius = T::from_f64(0.15 * (conf.x_max - conf.x_min)).unwrap(); // Smaller radius for better convergence
                let start_idx = i * pop_per_swarm;
                let mut swarm_pop: OMatrix<T, Dyn, D> =
                    init_pop.rows(start_idx, particles_per_swarm).into_owned();

                let particle_adjustments: Vec<_> = (0..particles_per_swarm)
                    .into_par_iter()
                    .map_init(
                        || rng.clone(),
                        |particle_rng, j| {
                            let mut particle_row = Vec::with_capacity(dim);
                            for k in 0..dim {
                                let r = T::from_f64(particle_rng.random::<f64>()).unwrap();
                                let noise = (r - T::from_f64(0.5).unwrap()) * radius;
                                let adjusted_value = (center[k] + noise).clamp(
                                    T::from_f64(conf.x_min).unwrap(),
                                    T::from_f64(conf.x_max).unwrap(),
                                );
                                particle_row.push(adjusted_value);
                            }
                            (j, particle_row)
                        },
                    )
                    .collect();

                for (j, particle_row) in particle_adjustments {
                    for (k, &value) in particle_row.iter().enumerate() {
                        swarm_pop[(j, k)] = value;
                    }
                }

                Swarm::new(
                    SwarmConfig {
                        num_particles: particles_per_swarm,
                        dim,
                        c1: T::from_f64(conf.c1).unwrap(),
                        c2: T::from_f64(conf.c2).unwrap(),
                        bounds: (
                            T::from_f64(conf.x_min).unwrap(),
                            T::from_f64(conf.x_max).unwrap(),
                        ),
                        opt_prob,
                        init_pop: swarm_pop,
                        inertia_start: T::from_f64(conf.inertia_start).unwrap(),
                        inertia_end: T::from_f64(conf.inertia_end).unwrap(),
                        max_iterations: max_iter,
                    },
                    seed,
                )
            },
        )
        .collect()
}

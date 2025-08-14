use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::algorithms::multi_swarm::swarm::{initialize_swarms, Swarm};
use crate::utils::config::MSPOConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

pub struct MSPO<T, N, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    N: Dim + Send + Sync,
    DefaultAllocator:
        Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D> + Allocator<U1>,
{
    pub conf: MSPOConf,
    pub st: State<T, N, D>,
    pub swarms: Vec<Swarm<T, D>>,
    pub opt_prob: OptProb<T, D>,
    stagnation_counter: usize,
    last_best_fitness: T,
    improvement_threshold: T,
    generation_improvements: VecDeque<f64>,
    success_history: VecDeque<bool>,
}

impl<T, N, D> MSPO<T, N, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    N: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N, D>
        + Allocator<N>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<U1>,
{
    pub fn new(
        conf: MSPOConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        max_iter: usize,
    ) -> Self {
        let dim = init_pop.ncols();
        let total_particles = init_pop.nrows();
        assert!(
            total_particles >= conf.num_swarms * conf.swarm_size,
            "Initial population size must be at least num_swarms * swarm_size"
        );

        let (best_x, best_fitness) = Self::find_best_solution(&init_pop, &opt_prob);

        let swarms = initialize_swarms(&conf, dim, &init_pop, &opt_prob, max_iter);
        let (fitness, constraints): (Vec<T>, Vec<bool>) = (0..init_pop.nrows())
            .into_par_iter()
            .map(|i| {
                let x = init_pop.row(i).transpose();
                let fit = opt_prob.evaluate(&x);
                let constr = opt_prob.is_feasible(&x);
                (fit, constr)
            })
            .unzip();

        let fitness =
            OVector::<T, N>::from_vec_generic(N::from_usize(init_pop.nrows()), U1, fitness);
        let constraints =
            OVector::<bool, N>::from_vec_generic(N::from_usize(init_pop.nrows()), U1, constraints);

        let st = State {
            best_x,
            best_f: best_fitness,
            pop: init_pop,
            fitness,
            constraints,
            iter: 1,
        };

        // Extract values before moving conf
        let improvement_threshold = T::from_f64(conf.improvement_threshold).unwrap();

        Self {
            conf,
            st,
            swarms,
            opt_prob,
            stagnation_counter: 0,
            last_best_fitness: best_fitness,
            improvement_threshold,
            generation_improvements: VecDeque::with_capacity(20),
            success_history: VecDeque::with_capacity(20),
        }
    }

    fn find_best_solution(
        population: &OMatrix<T, N, D>,
        opt_prob: &OptProb<T, D>,
    ) -> (OVector<T, D>, T) {
        (0..population.nrows())
            .filter_map(|i| {
                let x = population.row(i).transpose();
                if opt_prob.is_feasible(&x) {
                    Some((x.clone(), opt_prob.evaluate(&x)))
                } else {
                    None
                }
            })
            .max_by(|(_, f1), (_, f2)| f1.partial_cmp(f2).unwrap())
            .unwrap_or_else(|| {
                let x = population.row(0).transpose();
                (x.clone(), opt_prob.evaluate(&x))
            })
    }

    fn exchange_information(&mut self) {
        let best_positions: Vec<_> = self
            .swarms
            .iter()
            .map(|swarm| {
                (
                    swarm.global_best_position.clone(),
                    swarm.global_best_fitness,
                )
            })
            .collect();

        let mut swarm_indices: Vec<_> = (0..self.swarms.len()).collect();
        swarm_indices.sort_by(|&i, &j| {
            best_positions[i]
                .1
                .partial_cmp(&best_positions[j].1)
                .unwrap()
        });

        let exchange_ratio = self.compute_adaptive_exchange_ratio();

        self.swarms
            .par_iter_mut()
            .enumerate()
            .for_each(|(_swarm_idx, swarm)| {
                let better_swarms: Vec<_> = swarm_indices
                    .iter()
                    .filter(|&&idx| {
                        best_positions[idx].1 > swarm.global_best_fitness
                            && best_positions[idx].1
                                > swarm.global_best_fitness
                                    * (T::one() + self.improvement_threshold)
                    })
                    .collect();

                if !better_swarms.is_empty() {
                    let num_exchange = (self.conf.swarm_size as f64 * exchange_ratio) as usize;
                    let mut particles: Vec<_> = swarm.particles.iter_mut().enumerate().collect();

                    // Exchange worst particles first
                    particles.sort_by(|(_, p1), (_, p2)| {
                        p1.best_fitness.partial_cmp(&p2.best_fitness).unwrap()
                    });

                    // Take information from better swarms
                    for (_, particle) in particles.iter_mut().take(num_exchange) {
                        for &better_idx in &better_swarms {
                            let (better_pos, better_fitness) = &best_positions[*better_idx];

                            // Only update if better and feasible
                            if *better_fitness
                                > particle.best_fitness * (T::one() + self.improvement_threshold)
                                && self.opt_prob.is_feasible(better_pos)
                            {
                                particle.best_position = better_pos.clone();
                                particle.best_fitness = *better_fitness;
                                break; // Only use the first better solution
                            }
                        }
                    }
                }
            });
    }

    fn compute_adaptive_exchange_ratio(&self) -> f64 {
        let base_ratio = self.conf.exchange_ratio;

        // Increase exchange when stagnated
        if self.stagnation_counter > 10 {
            (base_ratio * 1.5).min(0.3)
        } else if self.stagnation_counter > 5 {
            (base_ratio * 1.2).min(0.25)
        } else {
            base_ratio
        }
    }

    fn check_stagnation(&mut self) {
        let current_fitness = self.st.best_f;
        let improvement = current_fitness - self.last_best_fitness;

        let improvement_f64 = improvement.to_f64().unwrap_or(0.0);
        self.generation_improvements.push_back(improvement_f64);
        if self.generation_improvements.len() > 20 {
            self.generation_improvements.pop_front();
        }

        let success = improvement > self.improvement_threshold;
        self.success_history.push_back(success);
        if self.success_history.len() > 20 {
            self.success_history.pop_front();
        }

        if improvement < self.improvement_threshold {
            self.stagnation_counter += 1;
        } else {
            self.stagnation_counter = 0;
        }

        self.last_best_fitness = current_fitness;
    }

    pub fn stagnation_counter(&self) -> usize {
        self.stagnation_counter
    }

    pub fn is_stagnated(&self) -> bool {
        self.stagnation_counter > 20
    }

    pub fn get_swarm_diversity(&self) -> Vec<f64> {
        self.swarms.iter().map(|s| s.current_diversity()).collect()
    }

    pub fn get_average_improvement(&self, window_size: usize) -> Vec<T> {
        self.swarms
            .iter()
            .map(|s| s.average_improvement(window_size))
            .collect()
    }

    pub fn get_performance_stats(&self) -> (f64, f64, f64) {
        let avg_improvement = if self.generation_improvements.len() > 5 {
            self.generation_improvements.iter().sum::<f64>()
                / self.generation_improvements.len() as f64
        } else {
            0.0
        };

        let success_rate = if !self.success_history.is_empty() {
            self.success_history.iter().filter(|&&x| x).count() as f64
                / self.success_history.len() as f64
        } else {
            0.0
        };

        let stagnation_level = self.stagnation_counter as f64;

        (avg_improvement, success_rate, stagnation_level)
    }

    pub fn get_population(&self) -> OMatrix<T, N, D> {
        let total_particles = self.swarms.len() * self.conf.swarm_size;
        let dim = self.st.best_x.len();
        let mut population =
            OMatrix::<T, N, D>::zeros_generic(N::from_usize(total_particles), D::from_usize(dim));

        let mut positions = Vec::with_capacity(total_particles);

        for (swarm_idx, swarm) in self.swarms.iter().enumerate() {
            for (particle_idx, particle) in swarm.particles.iter().enumerate() {
                let row = swarm_idx * self.conf.swarm_size + particle_idx;
                positions.push((row, particle.position.clone()));
            }
        }

        for (row, position) in positions {
            population.set_row(row, &position.transpose());
        }

        population
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for MSPO<T, N, D>
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N, D>
        + Allocator<N>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<U1>,
{
    fn step(&mut self) {
        let results: Vec<_> = self
            .swarms
            .par_iter_mut()
            .map(|swarm| {
                swarm.update(&self.opt_prob);
                (
                    swarm.global_best_position.clone(),
                    swarm.global_best_fitness,
                )
            })
            .collect();

        for (pos, fitness) in results {
            if fitness > self.st.best_f && self.opt_prob.is_feasible(&pos) {
                self.st.best_f = fitness;
                self.st.best_x = pos;
            }
        }

        self.check_stagnation();

        let exchange_interval = if self.stagnation_counter > 10 {
            self.conf.exchange_interval / 2 // More frequent exchange when stagnated
        } else {
            self.conf.exchange_interval
        };

        if self.st.iter % exchange_interval == 0 {
            self.exchange_information(); // Periodically exchange info
        }

        self.st.pop = self.get_population();

        let (fitness, constraints): (Vec<T>, Vec<bool>) = (0..self.st.pop.nrows())
            .into_par_iter()
            .map(|i| {
                let x = self.st.pop.row(i).transpose();
                let fit = self.opt_prob.evaluate(&x);
                let constr = self.opt_prob.is_feasible(&x);
                (fit, constr)
            })
            .unzip();

        self.st.fitness =
            OVector::<T, N>::from_vec_generic(N::from_usize(self.st.pop.nrows()), U1, fitness);
        self.st.constraints = OVector::<bool, N>::from_vec_generic(
            N::from_usize(self.st.pop.nrows()),
            U1,
            constraints,
        );
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

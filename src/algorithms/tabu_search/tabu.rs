use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::utils::alg_conf::tabu_conf::RestartStrategy;
use crate::utils::config::TabuConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::tabu_search::{
    neighborhood::{create_neighborhood_generator, NeighborhoodGenerator},
    tabu_list::{TabuList, TabuType},
};

pub struct TabuSearch<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: TabuConf,
    pub opt_prob: OptProb<T, D>,
    pub x: OVector<T, D>,
    pub st: State<T, N, D>,
    tabu_list: TabuList<T, D>,
    neighborhood_generator: NeighborhoodGenerator<T, D>,
    iterations_since_improvement: usize,
    success_history: VecDeque<bool>,
    improvement_history: VecDeque<f64>,
    current_step_size: f64,
    current_perturbation_prob: f64,
    phase: SearchPhase,
    phase_iterations: usize,
    rng: StdRng,
    seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
enum SearchPhase {
    Intensification,
    Diversification,
}

impl<T, N, D> TabuSearch<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<N>,
{
    pub fn new(
        conf: TabuConf,
        init_pop: OMatrix<T, U1, D>,
        opt_prob: OptProb<T, D>,
        seed: u64,
    ) -> Self {
        let init_x = init_pop.row(0).transpose();
        let best_f = opt_prob.evaluate(&init_x);
        let tabu_type = TabuType::from(&conf);
        let n = init_x.len();

        let neighborhood_generator =
            create_neighborhood_generator(&conf.advanced.neighborhood_strategy);

        Self {
            conf: conf.clone(),
            opt_prob: opt_prob.clone(),
            x: init_x.clone(),
            st: State {
                best_x: init_x.clone(),
                best_f,
                pop: OMatrix::<T, N, D>::from_fn_generic(
                    N::from_usize(1),
                    D::from_usize(n),
                    |_, j| init_x.clone()[j],
                ),
                fitness: OVector::<T, N>::from_element_generic(N::from_usize(1), U1, best_f),
                constraints: OVector::<bool, N>::from_element_generic(
                    N::from_usize(1),
                    U1,
                    opt_prob.is_feasible(&init_x.clone()),
                ),
                iter: 1,
            },
            tabu_list: TabuList::new(conf.common.tabu_list_size, tabu_type),
            neighborhood_generator,
            iterations_since_improvement: 0,
            success_history: VecDeque::with_capacity(conf.advanced.success_history_size),
            improvement_history: VecDeque::with_capacity(conf.advanced.success_history_size),
            current_step_size: conf.common.step_size,
            current_perturbation_prob: conf.common.perturbation_prob,
            phase: SearchPhase::Intensification,
            phase_iterations: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    fn generate_neighbor(&self, rng: &mut impl Rng) -> OVector<T, D> {
        self.neighborhood_generator.generate_neighbor(&self.x, rng)
    }

    fn evaluate_neighbor(&self, neighbor: &OVector<T, D>) -> Option<T> {
        if self.opt_prob.is_feasible(neighbor) {
            let is_tabu = self.tabu_list.is_tabu(
                neighbor,
                T::from_f64(self.conf.common.tabu_threshold).unwrap(),
            );

            if !is_tabu {
                Some(self.opt_prob.evaluate(neighbor))
            } else if self.conf.advanced.aspiration_criteria {
                let fitness = self.opt_prob.evaluate(neighbor);
                if self.tabu_list.can_aspire(neighbor, fitness) {
                    Some(fitness)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    fn track_success(&mut self, old_fitness: T, new_fitness: T) {
        if !self.conf.advanced.adaptive_parameters {
            return;
        }

        let improved = new_fitness > old_fitness;
        self.success_history.push_back(improved);

        if self.success_history.len() > self.conf.advanced.success_history_size {
            self.success_history.pop_front();
        }

        let improvement = (new_fitness - old_fitness).to_f64().unwrap_or(0.0);
        self.improvement_history.push_back(improvement);

        if self.improvement_history.len() > self.conf.advanced.success_history_size {
            self.improvement_history.pop_front();
        }
    }

    fn adapt_parameters(&mut self) {
        if !self.conf.advanced.adaptive_parameters {
            return;
        }

        if self.success_history.len() < 5 {
            return;
        }

        let success_rate = self.success_history.iter().filter(|&&x| x).count() as f64
            / self.success_history.len() as f64;

        let avg_improvement =
            self.improvement_history.iter().sum::<f64>() / self.improvement_history.len() as f64;

        if success_rate < 0.2 {
            self.current_step_size *= 1.0 + self.conf.advanced.adaptation_rate; // Low success - increase exploration
        } else if success_rate > 0.6 && avg_improvement > 1e-4 {
            self.current_step_size *= 1.0 - self.conf.advanced.adaptation_rate * 0.3;
            // High success - fine-tune
        }

        if success_rate < 0.2 {
            self.current_perturbation_prob *= 1.0 + self.conf.advanced.adaptation_rate * 0.5;
        } else if success_rate > 0.6 {
            self.current_perturbation_prob *= 1.0 - self.conf.advanced.adaptation_rate * 0.3;
        }

        self.current_step_size = self.current_step_size.clamp(
            self.conf.common.step_size * 0.1,
            self.conf.common.step_size * 10.0,
        );
        self.current_perturbation_prob = self.current_perturbation_prob.clamp(0.05, 0.8);

        self.neighborhood_generator
            .update_parameters(success_rate, avg_improvement);
    }

    fn check_restart(&mut self) -> bool {
        match &self.conf.advanced.restart_strategy {
            RestartStrategy::None => false,
            RestartStrategy::Periodic { frequency } => {
                self.st.iter > 0 && self.st.iter % frequency == 0
            }
            RestartStrategy::Stagnation {
                max_iterations,
                threshold,
            } => {
                self.iterations_since_improvement >= *max_iterations
                    && self
                        .improvement_history
                        .iter()
                        .take(10)
                        .all(|&x| x.abs() < *threshold)
            }
            RestartStrategy::Adaptive {
                base_frequency,
                adaptation_rate,
            } => {
                let adaptive_frequency = (*base_frequency as f64
                    * (1.0 + self.iterations_since_improvement as f64 * adaptation_rate))
                    as usize;
                self.st.iter > 0 && self.st.iter % adaptive_frequency == 0
            }
        }
    }

    // Reset when stagnant
    fn restart_search(&mut self) {
        self.tabu_list.clear_frequency_map();
        self.tabu_list.reset_quality_memory();

        self.success_history.clear();
        self.improvement_history.clear();

        self.current_step_size = self.conf.common.step_size;
        self.current_perturbation_prob = self.conf.common.perturbation_prob;

        // Generate a new random starting position for restart
        let mut local_rng = self.rng.clone();
        let neighbor = self.generate_neighbor(&mut local_rng);
        self.rng = local_rng;
        self.x = neighbor;

        // Switch phase
        self.phase = match self.phase {
            SearchPhase::Intensification => SearchPhase::Diversification,
            SearchPhase::Diversification => SearchPhase::Intensification,
        };
        self.phase_iterations = 0;
        self.iterations_since_improvement = 0;
    }

    fn update_search_phase(&mut self) {
        self.phase_iterations += 1;

        if self.phase_iterations >= self.conf.advanced.intensification_cycles {
            match self.phase {
                SearchPhase::Intensification => {
                    if self.iterations_since_improvement > 5 {
                        // harcoded at 5
                        self.phase = SearchPhase::Diversification;
                        self.phase_iterations = 0;
                    }
                }
                SearchPhase::Diversification => {
                    self.phase = SearchPhase::Intensification;
                    self.phase_iterations = 0;
                }
            }
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for TabuSearch<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<bool, N>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        let previous_best = self.st.best_f;

        if self.check_restart() {
            self.restart_search();
        }

        let mut best_neighbor = self.x.clone();
        let mut best_neighbor_fitness = T::neg_infinity();

        let neighbors: Vec<_> = (0..self.conf.common.num_neighbors)
            .into_par_iter()
            .map_init(
                || {
                    let thread_id = rayon::current_thread_index().unwrap_or(0);
                    StdRng::seed_from_u64(self.seed + self.st.iter as u64 * 1000 + thread_id as u64)
                },
                |rng, _| {
                    let neighbor = self.generate_neighbor(rng);
                    let fitness = self.evaluate_neighbor(&neighbor);
                    (neighbor, fitness)
                },
            )
            .filter_map(|(neighbor, fitness)| fitness.map(|f| (neighbor, f)))
            .collect();

        for (neighbor, fitness) in neighbors {
            if fitness > best_neighbor_fitness {
                best_neighbor = neighbor;
                best_neighbor_fitness = fitness;
            }
        }

        if best_neighbor_fitness > T::neg_infinity() {
            self.tabu_list.update(
                self.x.clone(),
                self.iterations_since_improvement,
                Some(best_neighbor_fitness),
            );

            self.x = best_neighbor.clone();

            if best_neighbor_fitness > self.st.best_f {
                self.st.best_f = best_neighbor_fitness;
                self.st.best_x = best_neighbor;
                self.iterations_since_improvement = 0;
            } else {
                self.iterations_since_improvement += 1;
            }
        }

        self.track_success(previous_best, self.st.best_f);
        self.adapt_parameters();
        self.update_search_phase();

        self.st.pop.row_mut(0).copy_from(&self.x.transpose());
        self.st.fitness[0] = self.opt_prob.evaluate(&self.x);
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

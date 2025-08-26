use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::utils::alg_conf::nm_conf::RestartStrategy;
use crate::utils::config::NelderMeadConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

pub struct NelderMead<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: NelderMeadConf,
    pub st: State<T, N, D>,
    pub opt_prob: OptProb<T, D>,
    pub simplex: Vec<OVector<T, D>>,
    current_alpha: f64,
    current_gamma: f64,
    current_rho: f64,
    current_sigma: f64,
    success_history: VecDeque<bool>,
    improvement_history: VecDeque<f64>,
    operation_success_counts: [usize; 4], // [reflection, expansion, contraction, shrink]
    stagnation_counter: usize,
    last_improvement: T,
    restart_counter: usize,
    last_restart_iter: usize,
    rng: StdRng,
}

impl<T, N, D> NelderMead<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<D, U1>,
{
    pub fn new(
        conf: NelderMeadConf,
        init_x: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        seed: u64,
    ) -> Self {
        let n: usize = init_x.ncols();
        assert_eq!(
            init_x.nrows(),
            n + 1,
            "Initial simplex must have n + 1 vertices"
        );

        let simplex: Vec<_> = (0..(n + 1)).map(|j| init_x.row(j).transpose()).collect();

        let fitness_values = OVector::<T, N>::from_iterator_generic(
            N::from_usize(n + 1),
            U1,
            simplex.iter().map(|vertex| opt_prob.evaluate(vertex)),
        );

        let best_idx = fitness_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let pop = OMatrix::<T, N, D>::from_iterator_generic(
            N::from_usize(n + 1),
            D::from_usize(n),
            simplex.iter().flat_map(|v| v.iter().cloned()),
        );

        let success_history_size = conf.advanced.success_history_size;
        let improvement_history_size = conf.advanced.improvement_history_size;

        Self {
            conf: conf.clone(),
            st: State {
                best_x: simplex[best_idx].clone(),
                best_f: fitness_values[best_idx],
                pop,
                fitness: fitness_values.clone(),
                constraints: OVector::<bool, N>::from_element_generic(
                    N::from_usize(n + 1),
                    U1,
                    true,
                ),
                iter: 1,
            },
            opt_prob,
            simplex: simplex.clone(),

            current_alpha: conf.common.alpha,
            current_gamma: conf.common.gamma,
            current_rho: conf.common.rho,
            current_sigma: conf.common.sigma,
            success_history: VecDeque::with_capacity(success_history_size),
            improvement_history: VecDeque::with_capacity(improvement_history_size),
            operation_success_counts: [0; 4],

            stagnation_counter: 0,
            last_improvement: fitness_values[best_idx],
            restart_counter: 0,
            last_restart_iter: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn centroid(&self, worst_idx: usize) -> OVector<T, D> {
        let mut centroid = OVector::<T, D>::zeros_generic(D::from_usize(self.st.best_x.len()), U1);
        for (i, vertex) in self.simplex.iter().enumerate() {
            if i != worst_idx {
                centroid += vertex;
            }
        }
        let scale = T::from_f64((self.simplex.len() - 1) as f64).unwrap();
        &centroid * (T::one() / scale)
    }

    fn get_sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.simplex.len()).collect();
        indices.sort_by(|&i, &j| {
            self.st.fitness[j]
                .partial_cmp(&self.st.fitness[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    // Reflect worst point across centroid
    fn try_reflection_expansion(
        &mut self,
        worst_idx: usize,
        best_idx: usize,
        centroid: &OVector<T, D>,
    ) -> bool {
        let reflected = centroid
            + (centroid - &self.simplex[worst_idx]) * T::from_f64(self.current_alpha).unwrap();
        let reflected_fitness = self.evaluate_point(&reflected);

        if reflected_fitness > self.st.fitness[worst_idx] {
            if reflected_fitness > self.st.fitness[best_idx] {
                let expanded =
                    centroid + (&reflected - centroid) * T::from_f64(self.current_gamma).unwrap();
                let expanded_fitness = self.evaluate_point(&expanded);

                if expanded_fitness > reflected_fitness {
                    self.update_vertex(worst_idx, expanded, expanded_fitness);
                    self.record_operation_success(2); // expansion
                    return true;
                } else {
                    self.update_vertex(worst_idx, reflected, reflected_fitness);
                    self.record_operation_success(0); // reflection
                    return true;
                }
            } else {
                self.update_vertex(worst_idx, reflected, reflected_fitness);
                self.record_operation_success(0); // reflection
                return true;
            }
        }

        false
    }

    fn try_contraction(
        &mut self,
        worst_idx: usize,
        _best_idx: usize,
        centroid: &OVector<T, D>,
    ) -> bool {
        let contracted = centroid
            + (&self.simplex[worst_idx] - centroid) * T::from_f64(self.current_rho).unwrap();
        let contracted_fitness = self.evaluate_point(&contracted);

        if contracted_fitness > self.st.fitness[worst_idx] {
            self.update_vertex(worst_idx, contracted, contracted_fitness);
            self.record_operation_success(1); // contraction
            return true;
        }

        false
    }

    // Close simplex around region
    fn shrink_simplex(&mut self, best_idx: usize) -> bool {
        let best = self.simplex[best_idx].clone();
        let shrink_results: Vec<_> = (0..self.simplex.len())
            .into_par_iter()
            .filter(|&i| i != best_idx)
            .map(|i| {
                let new_vertex =
                    &best + (&self.simplex[i] - &best) * T::from_f64(self.current_sigma).unwrap();
                let new_fitness = self.opt_prob.evaluate(&new_vertex);
                (i, new_vertex, new_fitness)
            })
            .collect();

        for (i, vertex, fitness) in shrink_results {
            self.update_vertex(i, vertex, fitness);
        }
        self.record_operation_success(3); // shrink
        true
    }

    // Negativ inf when infeasible
    fn evaluate_point(&self, point: &OVector<T, D>) -> T {
        if self.opt_prob.is_feasible(point) {
            self.opt_prob.evaluate(point)
        } else {
            T::neg_infinity()
        }
    }

    fn update_vertex(&mut self, idx: usize, vertex: OVector<T, D>, fitness: T) {
        self.simplex[idx] = vertex;
        self.st.fitness[idx] = fitness;
    }

    fn update_best_solution(&mut self) {
        let best_idx = self
            .st
            .fitness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let old_best_f = self.st.best_f;
        self.st.best_x = self.simplex[best_idx].clone();

        if self.st.fitness[best_idx] > self.st.best_f {
            self.st.best_f = self.st.fitness[best_idx];
            self.st.best_x = self.simplex[best_idx].clone();

            let improvement = self.st.best_f - old_best_f;
            self.improvement_history
                .push_back(improvement.to_f64().unwrap_or(0.0));
            if self.improvement_history.len() > self.conf.advanced.improvement_history_size {
                self.improvement_history.pop_front();
            }

            self.stagnation_counter = 0;
            self.last_improvement = self.st.best_f;
        } else {
            self.stagnation_counter += 1;
        }

        self.st.pop = OMatrix::<T, N, D>::from_iterator_generic(
            N::from_usize(self.simplex.len()),
            D::from_usize(self.st.best_x.len()),
            self.simplex.iter().flat_map(|v| v.iter().cloned()),
        );
    }

    // Adapt
    fn adapt_parameters(&mut self) {
        if !self.conf.advanced.adaptive_parameters {
            return;
        }

        if self.success_history.len() < 5 {
            return;
        }

        let success_rate = self.success_history.iter().filter(|&&x| x).count() as f64
            / self.success_history.len() as f64;

        let avg_improvement = if self.improvement_history.len() > 5 {
            self.improvement_history.iter().sum::<f64>() / self.improvement_history.len() as f64
        } else {
            0.0
        };

        let adaptation_rate = self.conf.advanced.adaptation_rate;

        // Alpha (reflection)
        if success_rate < 0.2 {
            self.current_alpha *= 1.0 + adaptation_rate; // Low success - increase exploration
        } else if success_rate > 0.6 && avg_improvement > 1e-4 {
            self.current_alpha *= 1.0 - adaptation_rate * 0.3; // High success - fine-tune
        }

        // Gamma (expansion)
        if success_rate < 0.2 {
            self.current_gamma *= 1.0 + adaptation_rate * 0.5;
        } else if success_rate > 0.6 {
            self.current_gamma *= 1.0 - adaptation_rate * 0.3;
        }

        // Rho (contraction)
        if success_rate < 0.2 {
            self.current_rho *= 1.0 - adaptation_rate * 0.3;
        } else if success_rate > 0.5 {
            self.current_rho *= 1.0 + adaptation_rate * 0.2;
        }

        // Sigma (shrink)
        if success_rate < 0.2 {
            self.current_sigma *= 1.0 - adaptation_rate * 0.3;
        } else if success_rate > 0.5 {
            self.current_sigma *= 1.0 + adaptation_rate * 0.2;
        }

        // Bounds
        let bounds = &self.conf.advanced.coefficient_bounds;
        self.current_alpha = self
            .current_alpha
            .clamp(bounds.alpha_bounds.0, bounds.alpha_bounds.1);
        self.current_gamma = self
            .current_gamma
            .clamp(bounds.gamma_bounds.0, bounds.gamma_bounds.1);
        self.current_rho = self
            .current_rho
            .clamp(bounds.rho_bounds.0, bounds.rho_bounds.1);
        self.current_sigma = self
            .current_sigma
            .clamp(bounds.sigma_bounds.0, bounds.sigma_bounds.1);
    }

    fn record_operation_success(&mut self, operation_idx: usize) {
        self.operation_success_counts[operation_idx] += 1;
        self.success_history.push_back(true);
        if self.success_history.len() > self.conf.advanced.success_history_size {
            self.success_history.pop_front();
        }
    }

    fn record_operation_failure(&mut self) {
        self.success_history.push_back(false);
        if self.success_history.len() > self.conf.advanced.success_history_size {
            self.success_history.pop_front();
        }
    }

    fn check_restart(&mut self) -> bool {
        match &self.conf.advanced.restart_strategy {
            RestartStrategy::None => false,
            RestartStrategy::Periodic { frequency } => {
                self.st.iter - self.last_restart_iter >= *frequency
            }
            RestartStrategy::Stagnation {
                max_iterations,
                threshold,
            } => {
                self.stagnation_counter >= *max_iterations
                    || self.last_improvement.to_f64().unwrap_or(0.0) < *threshold
            }
            RestartStrategy::Adaptive {
                base_frequency,
                adaptation_rate,
            } => {
                let adaptive_frequency = (*base_frequency as f64
                    * (1.0 + adaptation_rate * self.stagnation_counter as f64))
                    as usize;
                self.st.iter - self.last_restart_iter >= adaptive_frequency
            }
        }
    }

    fn perform_restart(&mut self) {
        let current_best = self.st.best_x.clone();
        let current_best_f = self.st.best_f;

        // New simplex around current best
        let simplex_size = self.simplex.len();
        let dim = current_best.len();

        let mut new_simplex = vec![current_best.clone()];

        for _i in 1..simplex_size {
            let mut new_vertex = current_best.clone();

            for j in 0..dim {
                let perturbation = T::from_f64(self.rng.random_range(-0.1..0.1)).unwrap();
                new_vertex[j] += perturbation;
            }

            new_vertex = self.project_to_bounds(new_vertex);
            new_simplex.push(new_vertex);
        }

        let fitness_values = OVector::<T, N>::from_iterator_generic(
            N::from_usize(simplex_size),
            U1,
            new_simplex
                .iter()
                .map(|vertex| self.opt_prob.evaluate(vertex)),
        );

        self.simplex = new_simplex;
        self.st.fitness = fitness_values;
        self.st.pop = OMatrix::<T, N, D>::from_iterator_generic(
            N::from_usize(simplex_size),
            D::from_usize(dim),
            self.simplex.iter().flat_map(|v| v.iter().cloned()),
        );

        // Reinit
        self.stagnation_counter = 0;
        self.last_improvement = current_best_f;
        self.restart_counter += 1;
        self.last_restart_iter = self.st.iter;
        self.current_alpha = self.conf.common.alpha;
        self.current_gamma = self.conf.common.gamma;
        self.current_rho = self.conf.common.rho;
        self.current_sigma = self.conf.common.sigma;
    }

    fn project_to_bounds(&self, point: OVector<T, D>) -> OVector<T, D> {
        if let (Some(lb), Some(ub)) = (
            self.opt_prob.objective.x_lower_bound(&point),
            self.opt_prob.objective.x_upper_bound(&point),
        ) {
            let mut projected = point;
            for i in 0..projected.len() {
                projected[i] = projected[i].max(lb[i]).min(ub[i]);
            }
            projected
        } else {
            point
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for NelderMead<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<D, U1>,
{
    fn step(&mut self) {
        if self.check_restart() {
            self.perform_restart();
        }

        let indices = self.get_sorted_indices();
        let (worst_idx, best_idx) = (indices[indices.len() - 1], indices[0]);

        let centroid = self.centroid(worst_idx);

        let operation_successful = self.try_reflection_expansion(worst_idx, best_idx, &centroid)
            || self.try_contraction(worst_idx, best_idx, &centroid)
            || self.shrink_simplex(best_idx);

        if !operation_successful {
            self.record_operation_failure();
        }

        self.update_best_solution();
        self.adapt_parameters();

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }

    fn get_simplex(&self) -> Option<&Vec<OVector<T, D>>> {
        Some(&self.simplex)
    }
}

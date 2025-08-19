use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::Rng;
use rayon::prelude::*;

use crate::utils::config::GRASPConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

pub struct GRASP<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: GRASPConf,
    pub st: State<T, N, D>,
    pub opt_prob: OptProb<T, D>,
    cached_lower_bounds: Option<OVector<T, D>>,
    cached_upper_bounds: Option<OVector<T, D>>,
    stagnation_count: usize,
    last_improvement: usize,
}

impl<T, N, D> GRASP<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<bool, N>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<N>,
{
    pub fn new(conf: GRASPConf, init_pop: OMatrix<T, U1, D>, opt_prob: OptProb<T, D>) -> Self {
        let init_x = init_pop.row(0).transpose();
        let best_f = opt_prob.evaluate(&init_x);
        let n = init_x.len();

        let (cached_lower_bounds, cached_upper_bounds) = if conf.cache_bounds {
            let lb = opt_prob.objective.x_lower_bound(&init_x);
            let ub = opt_prob.objective.x_upper_bound(&init_x);
            (lb, ub)
        } else {
            (None, None)
        };

        Self {
            conf,
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
            opt_prob,
            cached_lower_bounds,
            cached_upper_bounds,
            stagnation_count: 0,
            last_improvement: 0,
        }
    }

    fn get_bounds(&self, candidate: &OVector<T, D>) -> (OVector<T, D>, OVector<T, D>) {
        if let (Some(lb), Some(ub)) = (&self.cached_lower_bounds, &self.cached_upper_bounds) {
            (lb.clone(), ub.clone())
        } else {
            let lb = self
                .opt_prob
                .objective
                .x_lower_bound(candidate)
                .unwrap_or_else(|| {
                    OVector::<T, D>::from_element_generic(
                        D::from_usize(candidate.len()),
                        U1,
                        T::from_f64(-10.0).unwrap(),
                    )
                });
            let ub = self
                .opt_prob
                .objective
                .x_upper_bound(candidate)
                .unwrap_or_else(|| {
                    OVector::<T, D>::from_element_generic(
                        D::from_usize(candidate.len()),
                        U1,
                        T::from_f64(10.0).unwrap(),
                    )
                });
            (lb, ub)
        }
    }

    // Greedy randomized construction
    pub fn construct_solution(&self) -> OVector<T, D> {
        let candidates: Vec<OVector<T, D>> = (0..self.conf.num_candidates)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng(); // Create new RNG for each thread
                let mut candidate =
                    OVector::<T, D>::zeros_generic(D::from_usize(self.st.best_x.len()), U1);

                let (lb, ub) = self.get_bounds(&candidate);

                // Generate value within restricted candidate list (RCL)
                for i in 0..self.st.best_x.len() {
                    let adaptive_alpha = if self.stagnation_count > 5 {
                        (self.conf.alpha * 1.5).min(0.8)
                    } else {
                        self.conf.alpha
                    };

                    let alpha = T::from_f64(adaptive_alpha).unwrap();
                    let range = ub[i] - lb[i];
                    let rcl_min = lb[i] + (T::one() - alpha) * range;
                    let rcl_max = ub[i] - (T::one() - alpha) * range;

                    let (min_val, max_val) = if rcl_min < rcl_max {
                        (rcl_min, rcl_max)
                    } else if (rcl_min - rcl_max).abs() < T::from_f64(1e-10).unwrap() {
                        // If very close, use small range around the value
                        let epsilon = T::from_f64(1e-6).unwrap();
                        (rcl_min - epsilon, rcl_max + epsilon)
                    } else {
                        // If rcl_min > rcl_max, use full bounds
                        eprintln!("Warning: Invalid RCL bounds for dimension {}: lb[{}]={:?}, ub[{}]={:?}, alpha={}, rcl_min={:?}, rcl_max={:?}", 
                                 i, i, lb[i], i, ub[i], adaptive_alpha, rcl_min, rcl_max);
                        (lb[i], ub[i])
                    };

                    candidate[i] = T::from_f64(
                        rng.random_range(min_val.to_f64().unwrap()..max_val.to_f64().unwrap()),
                    )
                    .unwrap();
                }
                candidate
            })
            .collect();

        // Select best feasible candidate
        candidates
            .into_iter()
            .filter(|c| self.opt_prob.is_feasible(c))
            .max_by(|a, b| {
                let fa = self.opt_prob.evaluate(a);
                let fb = self.opt_prob.evaluate(b);
                fa.partial_cmp(&fb).unwrap()
            })
            .unwrap_or(self.st.best_x.clone())
    }

    // Local search phase
    pub fn local_search(&self, solution: &OVector<T, D>) -> OVector<T, D> {
        let mut current = solution.clone();
        let mut current_fitness = self.opt_prob.evaluate(&current);
        let mut improved = true;
        let mut local_iter = 0;

        let adaptive_step_size = if self.stagnation_count > 5 {
            self.conf.step_size * 3.0 // Much bigger step size when stuck
        } else {
            self.conf.step_size
        };

        let adaptive_perturbation_prob = if self.stagnation_count > 5 {
            (self.conf.perturbation_prob * 2.0).min(0.9) // Much bigger perturbation when stuck
        } else {
            self.conf.perturbation_prob
        };

        while improved && local_iter < self.conf.max_local_iter {
            improved = false;
            local_iter += 1;

            // Generate and evaluate neighborhood in parallel
            let neighbors: Vec<OVector<T, D>> = (0..self.conf.num_neighbors)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::rng();
                    let mut neighbor = current.clone();

                    // Perturb random dimensions
                    for i in 0..neighbor.len() {
                        if rng.random_bool(adaptive_perturbation_prob) {
                            neighbor[i] += T::from_f64(
                                rng.random_range(-adaptive_step_size..adaptive_step_size),
                            )
                            .unwrap();
                        }
                    }

                    let (lb, ub) = self.get_bounds(&neighbor);
                    for i in 0..neighbor.len() {
                        neighbor[i] = neighbor[i].max(lb[i]).min(ub[i]);
                    }

                    neighbor
                })
                .collect();

            // Find best feasible neighbor
            if let Some(best_neighbor) = neighbors
                .into_iter()
                .filter(|n| self.opt_prob.is_feasible(n))
                .max_by(|a, b| {
                    let fa = self.opt_prob.evaluate(a);
                    let fb = self.opt_prob.evaluate(b);
                    fa.partial_cmp(&fb).unwrap()
                })
            {
                let neighbor_fitness = self.opt_prob.evaluate(&best_neighbor);
                if neighbor_fitness > current_fitness {
                    current = best_neighbor;
                    current_fitness = neighbor_fitness;
                    improved = true;
                }
            }
        }

        current
    }

    // Inject diversity to escape local optima
    fn add_diversity(&self, solution: &OVector<T, D>) -> OVector<T, D> {
        if self.stagnation_count > 5 {
            let mut rng = rand::rng();
            let mut diverse_solution = solution.clone();

            for i in 0..diverse_solution.len() {
                if rng.random_bool(self.conf.diversity_prob) {
                    let perturbation = T::from_f64(
                        rng.random_range(-3.0..3.0)
                            * self.conf.step_size
                            * self.conf.diversity_strength,
                    )
                    .unwrap();
                    diverse_solution[i] += perturbation;
                }
            }

            let (lb, ub) = self.get_bounds(&diverse_solution);
            for i in 0..diverse_solution.len() {
                diverse_solution[i] = diverse_solution[i].max(lb[i]).min(ub[i]);
            }

            diverse_solution
        } else {
            solution.clone()
        }
    }

    fn should_restart(&self) -> bool {
        self.stagnation_count > self.conf.restart_threshold
    }

    fn restart(&mut self) {
        let mut rng = rand::rng();
        let (lb, ub) = self.get_bounds(&self.st.best_x);

        // Generate a completely new random solution
        let mut new_solution =
            OVector::<T, D>::zeros_generic(D::from_usize(self.st.best_x.len()), U1);

        for i in 0..new_solution.len() {
            new_solution[i] =
                T::from_f64(rng.random_range(lb[i].to_f64().unwrap()..ub[i].to_f64().unwrap()))
                    .unwrap();
        }

        self.st.pop.row_mut(0).copy_from(&new_solution.transpose());
        self.st.fitness[0] = self.opt_prob.evaluate(&new_solution);
        self.st.constraints[0] = self.opt_prob.is_feasible(&new_solution);

        self.stagnation_count = 0;

        eprintln!(
            "GRASP restart triggered after {} iterations without improvement",
            self.st.iter - self.last_improvement
        );
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for GRASP<T, N, D>
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
        if self.should_restart() {
            self.restart();
            return;
        }

        let solution = self.construct_solution();
        let diverse_solution = self.add_diversity(&solution);
        let improved_solution = self.local_search(&diverse_solution);

        let fitness = self.opt_prob.evaluate(&improved_solution);
        if fitness > self.st.best_f && self.opt_prob.is_feasible(&improved_solution) {
            self.st.best_f = fitness;
            self.st.best_x = improved_solution.clone();
            self.last_improvement = self.st.iter;
            self.stagnation_count = 0;
        } else {
            self.stagnation_count += 1;
        }

        self.st
            .pop
            .row_mut(0)
            .copy_from(&improved_solution.transpose());
        self.st.fitness[0] = fitness;
        self.st.constraints[0] = self.opt_prob.is_feasible(&improved_solution);

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::Rng;
use rayon::prelude::*;

use crate::algorithms::parallel_tempering::{
    metropolis_hastings::MetropolisHastings,
    replica_exchange::{Always, Periodic, Stochastic, SwapCheck},
};
use crate::utils::alg_conf::pt_conf::UpdateConf;
use crate::utils::config::{PTConf, SwapConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

pub struct PT<T, N, D>
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<D, D> + Allocator<U1, D>,
{
    pub conf: PTConf,
    pub metropolis_hastings: MetropolisHastings<T, D>,
    pub swap_check: SwapCheck,
    pub p_schedule: Vec<T>,
    pub population: Vec<OMatrix<T, N, D>>, // For each replica
    pub fitness: Vec<OVector<T, N>>,
    pub constraints: Vec<OVector<bool, N>>,
    pub opt_prob: OptProb<T, D>,
    pub best_individual: OVector<T, D>,
    pub best_fitness: T,
    pub step_sizes: Vec<Vec<OMatrix<T, D, D>>>,
    pub st: State<T, N, D>,
}

impl<T, N, D> PT<T, N, D>
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<D, D> + Allocator<U1, D>,
{
    pub fn new(
        conf: PTConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        max_iter: usize,
    ) -> Self {

        assert!(
            conf.common.num_replicas > 0,
            "Number of replicas must be positive"
        );
        assert!(max_iter > 0, "Maximum iterations must be positive");
        assert!(
            init_pop.nrows() > 0,
            "Initial population must have at least one individual"
        );
        assert!(
            init_pop.ncols() > 0,
            "Initial population must have at least one dimension"
        );

        let swap_check = match &conf.swap_conf {
            SwapConf::Periodic(p) => SwapCheck::Periodic(Periodic::new(p.swap_frequency, max_iter)),
            SwapConf::Stochastic(s) => SwapCheck::Stochastic(Stochastic::new(s.swap_probability)),
            SwapConf::Always(_) => SwapCheck::Always(Always::new()),
        };

        let metropolis_hastings = MetropolisHastings::<T, D>::new(
            opt_prob.clone(),
            &conf.update_conf,
            init_pop.row(0).transpose(),
        );

        // Power law schedule for cyclic annealing
        let p_init = conf.common.power_law_init;
        let p_final = conf.common.power_law_final;
        let p_cycles = conf.common.power_law_cycles;
        let num_iters = max_iter;

        let p_schedule: Vec<T> = (0..=num_iters)
            .map(|i| {
                let x = 2.0
                    * std::f64::consts::PI
                    * (p_cycles as f64 + 0.5)
                    * (i as f64 / num_iters as f64);
                let p_current = p_init + (p_final - p_init) * 0.5 * (1.0 - x.cos());
                T::from_f64(p_current).unwrap()
            })
            .collect();

        // Init populations sequentially
        let mut population = Vec::with_capacity(conf.common.num_replicas);
        let mut fitness = Vec::with_capacity(conf.common.num_replicas);
        let mut constraints = Vec::with_capacity(conf.common.num_replicas);

        for _ in 0..conf.common.num_replicas {
            let mut pop = OMatrix::<T, N, D>::zeros_generic(
                N::from_usize(init_pop.nrows()),
                D::from_usize(init_pop.ncols()),
            );
            for i in 0..init_pop.nrows() {
                pop.set_row(i, &init_pop.row(i));
            }

            let fit: Vec<T> = (0..init_pop.nrows())
                .map(|i| {
                    let individual = init_pop.row(i).transpose();
                    opt_prob.evaluate(&individual)
                })
                .collect();

            let constr: Vec<bool> = (0..init_pop.nrows())
                .map(|i| {
                    let individual = init_pop.row(i).transpose();
                    opt_prob.is_feasible(&individual)
                })
                .collect();

            population.push(pop);
            fitness.push(OVector::<T, N>::from_vec_generic(
                N::from_usize(init_pop.nrows()),
                U1,
                fit,
            ));
            constraints.push(OVector::<bool, N>::from_vec_generic(
                N::from_usize(init_pop.nrows()),
                U1,
                constr,
            ));
        }

        // Find best individual across all replicas
        let mut best_idx = 0;
        let mut best_row = 0;
        let mut best_fitness = fitness[0][0];
        for i in 0..conf.common.num_replicas {
            for j in 0..fitness[i].len() {
                if fitness[i][j] > best_fitness && constraints[i][j] {
                    best_fitness = fitness[i][j];
                    best_idx = i;
                    best_row = j;
                }
            }
        }

        let best_individual = population[best_idx].row(best_row).transpose().into_owned();

        // Initialize step sizes for each replica and individual
        let step_size_value = match &conf.update_conf {
            UpdateConf::MetropolisHastings(conf) => conf.random_walk_step_size,
            UpdateConf::MALA(conf) => conf.step_size,
            UpdateConf::PCN(conf) => conf.step_size,
            UpdateConf::Auto(_) => conf.common.mala_step_size,
        };

        let step_sizes: Vec<Vec<OMatrix<T, D, D>>> = (0..conf.common.num_replicas)
            .map(|_| {
                (0..population[0].nrows())
                    .map(|_| {
                        OMatrix::<T, D, D>::identity_generic(
                            D::from_usize(population[0].ncols()),
                            D::from_usize(population[0].ncols()),
                        ) * T::from_f64(step_size_value).unwrap()
                    })
                    .collect()
            })
            .collect();

        Self {
            conf,
            metropolis_hastings,
            swap_check,
            p_schedule,
            population: population.clone(),
            fitness: fitness.clone(),
            constraints: constraints.clone(),
            opt_prob,
            best_individual: best_individual.clone(),
            best_fitness,
            step_sizes,
            st: State {
                best_x: best_individual,
                best_f: best_fitness,
                pop: population[0].clone(),
                fitness: fitness[0].clone(),
                constraints: constraints[0].clone(),
                iter: 1,
            },
        }
    }

    // Temperature should be in [0, 1] range, with replica 0 being coldest (0) and highest being hottest (1)
    fn get_temperature(&self, replica_idx: usize) -> T {
        let power = self.p_schedule[self.st.iter - 1].to_f64().unwrap();
        let temp = (replica_idx as f64 / (self.conf.common.num_replicas - 1) as f64).powf(power);
        T::from_f64(temp).unwrap()
    }

    pub fn swap(&mut self) {
        let n = self.population.len();
        let m = self.population[0].nrows();

        // Try swapping adjacent replicas
        for i in 0..n - 1 {
            let t_i = self.get_temperature(i);
            let t_j = self.get_temperature(i + 1);

            // Calculate ensemble-level energy difference
            let mut total_energy_i = T::zero();
            let mut total_energy_j = T::zero();
            
            for k in 0..m {
                total_energy_i += self.fitness[i][k];
                total_energy_j += self.fitness[i + 1][k];
            }

            // Standard replica exchange acceptance criterion
            let delta_beta = (T::from_f64(1.0).unwrap() / t_j) - (T::from_f64(1.0).unwrap() / t_i);
            let delta_e = total_energy_j - total_energy_i;

            let log_acceptance = -delta_beta * delta_e;
            let acceptance_prob = if log_acceptance > T::from_f64(0.0).unwrap() {
                T::from_f64(1.0).unwrap()
            } else {
                log_acceptance.exp()
            };

            // Accept/reject the entire replica swap based on ensemble-level criterion
            if rand::rng().random::<f64>() < acceptance_prob.to_f64().unwrap() {
                self.population.swap(i, i + 1);
                self.fitness.swap(i, i + 1);
                self.constraints.swap(i, i + 1);
                self.step_sizes.swap(i, i + 1);
            }
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for PT<T, N, D>
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator:
        Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<D, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        let current_power = self.p_schedule[self.st.iter - 1].to_f64().unwrap();

        let temperatures: Vec<T> = (0..self.conf.common.num_replicas)
            .map(|k| {
                let power = current_power;
                T::from_f64((k as f64 / self.conf.common.num_replicas as f64).powf(power)).unwrap()
            })
            .collect();

        let results: Vec<_> = (0..self.conf.common.num_replicas)
            .into_par_iter()
            .map(|i| {
                let mut local_population = self.population[i].clone();
                let mut local_fitness = self.fitness[i].clone();
                let mut local_constraints = self.constraints[i].clone();
                let mut local_step_sizes = self.step_sizes[i].clone();
                let mut local_metropolis_hastings = self.metropolis_hastings.clone();
                
                for j in 0..local_population.nrows() {
                    let x_old = local_population.row(j).transpose();
                    let x_new = local_metropolis_hastings.local_move(
                        &x_old,
                        &local_step_sizes[j],
                        temperatures[i],
                    );
                    
                    let fitness_new = self.opt_prob.evaluate(&x_new);
                    let constr_new = self.opt_prob.is_feasible(&x_new);
                    
                    if local_metropolis_hastings.accept_reject(
                        &x_old,
                        &x_new,
                        constr_new,
                        temperatures[i],
                        -T::from_f64(1.0).unwrap(), // Negative to signal local move
                    ) {
                        local_population.row_mut(j).copy_from(&x_new.transpose());
                        local_fitness[j] = fitness_new;
                        local_constraints[j] = constr_new;
                        
                        let new_step_size = local_metropolis_hastings.update_step_size(
                            &local_step_sizes[j],
                            T::from_f64(0.5).unwrap(), // Placeholder acceptance rate
                            temperatures[i],
                        );
                        local_step_sizes[j] = new_step_size;
                    }
                }
                
                (i, local_population, local_fitness, local_constraints, local_step_sizes)
            })
            .collect();
        
        for (i, pop, fit, constr, step) in results {
            self.population[i] = pop;
            self.fitness[i] = fit;
            self.constraints[i] = constr;
            self.step_sizes[i] = step;
        }

        if match &self.swap_check {
            SwapCheck::Periodic(p) => p.should_swap(self.st.iter),
            SwapCheck::Stochastic(s) => s.should_swap(self.st.iter),
            SwapCheck::Always(a) => a.should_swap(self.st.iter),
        } {
            self.swap();
        }

        // Update best individual across all replicas and iterations
        let best_result = (0..self.conf.common.num_replicas)
            .into_par_iter()
            .flat_map(|i| {
                (0..self.fitness[i].len())
                    .into_par_iter()
                    .filter_map(|j| {
                        if self.fitness[i][j] > self.best_fitness && self.constraints[i][j] {
                            Some((i, j, self.fitness[i][j]))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Update global best if we found a better solution
        if let Some((best_idx, best_row, best_fitness)) = best_result {
            if best_fitness > self.best_fitness {
                self.best_fitness = best_fitness;
                self.best_individual = self.population[best_idx]
                    .row(best_row)
                    .transpose()
                    .into_owned();
            }
        }



        // Update state with current best replica (replica 0)
        self.st.best_x = self.best_individual.clone();
        self.st.best_f = self.best_fitness;
        self.st.pop = self.population[0].clone();
        self.st.fitness = self.fitness[0].clone();
        self.st.constraints = self.constraints[0].clone();

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

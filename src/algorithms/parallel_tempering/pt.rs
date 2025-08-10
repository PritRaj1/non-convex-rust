use nalgebra::{allocator::Allocator, DMatrix, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::Rng;
use rayon::prelude::*;

use crate::utils::config::{PTConf, SwapConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::parallel_tempering::{
    metropolis_hastings::MetropolisHastings,
    replica_exchange::{Always, Periodic, Stochastic, SwapCheck},
};

// Type aliases to reduce complexity
type InitResult<T, N, D> = (OMatrix<T, N, D>, OVector<T, N>, OVector<bool, N>);
type UpdateResult<T, D> = Option<(OVector<T, D>, T, bool, OMatrix<T, D, D>)>;

pub struct PT<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<D, D>,
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
    pub st: State<T, N, D>, // Store a copy of final replica's population and fitness values
}

impl<T, N, D> PT<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<D, D>,
{
    pub fn new(
        conf: PTConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        max_iter: usize,
    ) -> Self {
        let swap_check = match &conf.swap_conf {
            SwapConf::Periodic(p) => SwapCheck::Periodic(Periodic::new(p.swap_frequency, max_iter)),
            SwapConf::Stochastic(s) => SwapCheck::Stochastic(Stochastic::new(s.swap_probability)),
            SwapConf::Always(_) => SwapCheck::Always(Always::new()),
        };

        let metropolis_hastings = MetropolisHastings::<T, D>::new(
            opt_prob.clone(),
            T::from_f64(conf.common.mala_step_size).unwrap(),
            T::from_f64(conf.common.alpha).unwrap(),
            T::from_f64(conf.common.omega).unwrap(),
            init_pop.row(0).transpose(),
        );

        // Power law schedule for cyclic annealing
        let p_init = conf.common.power_law_init;
        let p_final = conf.common.power_law_final;
        let p_cycles = conf.common.power_law_cycles;
        let num_iters = max_iter;

        let x: Vec<f64> = (0..=num_iters)
            .map(|i| {
                2.0 * std::f64::consts::PI * (p_cycles as f64 + 0.5) * (i as f64 / num_iters as f64)
            })
            .collect();

        let p_schedule: Vec<T> = x
            .iter()
            .map(|&xi| T::from_f64(p_init + (p_final - p_init) * 0.5 * (1.0 - xi.cos())).unwrap())
            .collect();

        // Initialize populations in parallel
        let init_results: Vec<InitResult<T, N, D>> = (0..conf.common.num_replicas)
            .into_par_iter()
            .map(|_| {
                let mut pop = OMatrix::<T, N, D>::zeros_generic(
                    N::from_usize(init_pop.nrows()),
                    D::from_usize(init_pop.ncols()),
                );
                for i in 0..init_pop.nrows() {
                    pop.set_row(i, &init_pop.row(i));
                }

                let fit: Vec<T> = (0..init_pop.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let individual = init_pop.row(i).transpose();
                        opt_prob.evaluate(&individual)
                    })
                    .collect();

                let constr: Vec<bool> = (0..init_pop.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let individual = init_pop.row(i).transpose();
                        opt_prob.is_feasible(&individual)
                    })
                    .collect();

                (
                    pop,
                    OVector::<T, N>::from_vec_generic(N::from_usize(init_pop.nrows()), U1, fit),
                    OVector::<bool, N>::from_vec_generic(
                        N::from_usize(init_pop.nrows()),
                        U1,
                        constr,
                    ),
                )
            })
            .collect();

        // Unzip the results
        let mut population = Vec::with_capacity(conf.common.num_replicas);
        let mut fitness = Vec::with_capacity(conf.common.num_replicas);
        let mut constraints = Vec::with_capacity(conf.common.num_replicas);

        for (pop, fit, constr) in init_results {
            population.push(pop);
            fitness.push(fit);
            constraints.push(constr);
        }

        // Find best individual across all replicas
        let mut best_idx = 0;
        let mut best_fitness = fitness[0][0];
        for i in 0..conf.common.num_replicas {
            for j in 0..fitness[i].len() {
                if fitness[i][j] > best_fitness && constraints[i][j] {
                    best_fitness = fitness[i][j];
                    best_idx = i;
                }
            }
        }

        let best_individual = population[best_idx].row(0).transpose().into_owned();
        let step_sizes: Vec<Vec<OMatrix<T, D, D>>> = (0..conf.common.num_replicas)
            .map(|_| {
                (0..population[0].nrows())
                    .map(|_| {
                        OMatrix::<T, D, D>::identity_generic(
                            D::from_usize(population[0].ncols()),
                            D::from_usize(population[0].ncols()),
                        )
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

    // Replica exchange
    pub fn swap(&mut self) {
        let n = self.population.len();
        let m = self.population[0].nrows();

        // Initialize swap matrix for all possible pairs
        let mut swap_bool = DMatrix::from_element(n, n, false);

        // Randomly pick pairs to swap
        let mut rng = rand::rng();
        let num_attempts = n / 2;
        let swap_pairs: Vec<(usize, usize)> = (0..num_attempts)
            .map(|_| {
                let i = rng.random_range(0..n - 1);
                let j = rng.random_range(i + 1..n);
                (i, j)
            })
            .collect();

        // Determine which pairs to swap with mh criterion
        let swap_results: Vec<Vec<(usize, usize, usize, bool)>> = swap_pairs
            .par_iter()
            .map(|&(i, j)| {
                let power = self.p_schedule[self.st.iter].to_f64().unwrap();
                let t_i =
                    T::from_f64((i as f64 / self.conf.common.num_replicas as f64).powf(power))
                        .unwrap();
                let t_j =
                    T::from_f64((j as f64 / self.conf.common.num_replicas as f64).powf(power))
                        .unwrap();

                (0..m)
                    .into_par_iter()
                    .map(|k| {
                        let x_old = self.population[i].row(k).transpose();
                        let x_new = self.population[j].row(k).transpose();
                        let constraints_new = self.constraints[j][k];

                        let accept = self.metropolis_hastings.accept_reject(
                            &x_old,
                            &x_new,
                            constraints_new,
                            t_i,
                            t_j,
                        );

                        (i, j, k, accept)
                    })
                    .collect()
            })
            .collect();

        // Apply random factors after parallel section
        for swap_result in &swap_results {
            for (i, j, _, accept) in swap_result {
                if *accept {
                    // Add distance penalty to acceptance probability
                    let dist_factor = 0.9 + 0.1 * (1.0 - (*j - *i) as f64 / n as f64);
                    let accept_with_penalty = rng.random::<f64>() < dist_factor;

                    swap_bool[(*i, *j)] = swap_bool[(*i, *j)] || accept_with_penalty;
                    swap_bool[(*j, *i)] = swap_bool[(*i, *j)];
                }
            }
        }

        // Perform swaps
        let mut new_population = self.population.clone();
        let mut new_fitness = self.fitness.clone();
        let mut new_constraints = self.constraints.clone();

        // Process accepted swaps
        for i in 0..n {
            for j in (i + 1)..n {
                if swap_bool[(i, j)] {
                    for k in 0..m {
                        // Swap population rows
                        let temp_row = self.population[i].row(k);
                        new_population[i].set_row(k, &self.population[j].row(k));
                        new_population[j].set_row(k, &temp_row);

                        // Swap fitness values
                        let temp_fit = self.fitness[i][k];
                        new_fitness[i][k] = self.fitness[j][k];
                        new_fitness[j][k] = temp_fit;

                        // Swap constraints
                        let temp_const = self.constraints[i][k];
                        new_constraints[i][k] = self.constraints[j][k];
                        new_constraints[j][k] = temp_const;
                    }
                }
            }
        }

        self.population = new_population;
        self.fitness = new_fitness;
        self.constraints = new_constraints;
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for PT<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator:
        Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<D, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        let temperatures: Vec<T> = (0..self.conf.common.num_replicas)
            .map(|k| {
                let power = self.p_schedule[self.st.iter].to_f64().unwrap();
                T::from_f64((k as f64 / self.conf.common.num_replicas as f64).powf(power)).unwrap()
            })
            .collect();

        // Local move
        let updates: Vec<Vec<UpdateResult<T, D>>> = (0..self.conf.common.num_replicas)
            .into_par_iter()
            .map(|i| {
                (0..self.population[i].nrows())
                    .into_par_iter()
                    .map(|j| {
                        let x_old = self.population[i].row(j).transpose();
                        let x_new = self.metropolis_hastings.local_move(
                            &x_old,
                            &self.step_sizes[i][j],
                            temperatures[i],
                        );
                        let constr_new = self.opt_prob.is_feasible(&x_new);

                        if self.metropolis_hastings.accept_reject(
                            &x_old,
                            &x_new,
                            constr_new,
                            temperatures[i],
                            -T::from_f64(1.0).unwrap(), // Send in negative to signal local move
                        ) {
                            let new_step_size =
                                if self.opt_prob.objective.gradient(&x_old).is_none() {
                                    self.metropolis_hastings.update_step_size(
                                        &self.step_sizes[i][j],
                                        &x_old,
                                        &x_new,
                                    )
                                } else {
                                    self.step_sizes[i][j].clone()
                                };

                            let fitness_new = self.opt_prob.evaluate(&x_new);

                            Some((x_new.clone(), fitness_new, constr_new, new_step_size))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        // Apply updates
        for (i, replica) in updates.iter().enumerate() {
            for (j, update) in replica.iter().enumerate() {
                if let Some((x_new, fitness_new, constr_new, step_size_new)) = update {
                    self.population[i].row_mut(j).copy_from(&x_new.transpose());
                    self.fitness[i][j] = *fitness_new;
                    self.constraints[i][j] = *constr_new;
                    self.step_sizes[i][j] = step_size_new.clone();
                }
            }
        }

        // Replica exchange
        if match &self.swap_check {
            SwapCheck::Periodic(p) => p.should_swap(self.st.iter),
            SwapCheck::Stochastic(s) => s.should_swap(self.st.iter),
            SwapCheck::Always(a) => a.should_swap(self.st.iter),
        } {
            self.swap();
        }

        // Update best individual
        let mut best_idx = 0;
        let mut best_row = 0;
        let mut best_fitness = self.fitness[0][0];
        for i in 0..self.conf.common.num_replicas {
            for j in 0..self.fitness[i].len() {
                if self.fitness[i][j] > best_fitness && self.constraints[i][j] {
                    best_fitness = self.fitness[i][j];
                    best_idx = i;
                    best_row = j;
                }
            }
        }
        let best_individual = self.population[best_idx]
            .row(best_row)
            .transpose()
            .into_owned();
        self.best_individual = best_individual.clone();
        self.best_fitness = best_fitness;

        self.st.best_x = self.best_individual.clone();
        self.st.best_f = best_fitness;
        self.st.pop = self.population[0].clone();
        self.st.fitness = self.fitness[0].clone();
        self.st.constraints = self.constraints[0].clone();

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

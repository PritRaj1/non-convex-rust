use nalgebra::{
    allocator::Allocator, ComplexField, DefaultAllocator, Dim, DimSub, OMatrix, OVector, RealField,
    U1,
};

use std::iter::Sum;

use rand::Rng;
use rayon::prelude::*;

use crate::algorithms::parallel_tempering::{
    metropolis_hastings::MetropolisHastings,
    preconditioners::{Preconditioner, SampleCovariance},
    replica_exchange::{Always, Periodic, Stochastic, SwapCheck},
};
use crate::utils::alg_conf::pt_conf::UpdateConf;
use crate::utils::config::{PTConf, SwapConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

pub struct PT<T, N, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync + DimSub<nalgebra::Const<1>>,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N, D>
        + Allocator<N>
        + Allocator<D, D>
        + Allocator<U1, D>
        + Allocator<<D as DimSub<nalgebra::Const<1>>>::Output>,
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
    pub preconditioner: Box<dyn Preconditioner<T, N, D> + Send + Sync>,
    pub covariance_matrices: Vec<OMatrix<T, D, D>>,
    pub st: State<T, N, D>,
    acceptance_rates: Vec<f64>,
    swap_acceptance_rates: Vec<f64>,
    last_covariance_update: usize,
}

impl<T, N, D> PT<T, N, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync + DimSub<nalgebra::Const<1>>,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N, D>
        + Allocator<N>
        + Allocator<D, D>
        + Allocator<U1, D>
        + Allocator<<D as DimSub<nalgebra::Const<1>>>::Output>,
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

        let preconditioner: Box<dyn Preconditioner<T, N, D> + Send + Sync> =
            Box::new(SampleCovariance::new(T::from_f64(0.01).unwrap()));

        let mut covariance_matrices = Vec::with_capacity(conf.common.num_replicas);
        for i in 0..conf.common.num_replicas {
            let cov =
                preconditioner.compute_covariance(&population[i], &fitness[i], &constraints[i]);
            covariance_matrices.push(cov);
        }

        let num_replicas = conf.common.num_replicas;

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
            preconditioner,
            covariance_matrices,
            st: State {
                best_x: best_individual,
                best_f: best_fitness,
                pop: population[0].clone(),
                fitness: fitness[0].clone(),
                constraints: constraints[0].clone(),
                iter: 1,
            },
            acceptance_rates: vec![0.5; num_replicas],
            swap_acceptance_rates: vec![0.3; num_replicas.saturating_sub(1)],
            last_covariance_update: 0,
        }
    }

    // Temperature should be in [0, 1] range, with replica 0 being hottest (0) and highest being coldest (1)
    fn get_temperature(&self, replica_idx: usize) -> T {
        let schedule_idx = (self.st.iter - 1).min(self.p_schedule.len() - 1);
        let power = self.p_schedule[schedule_idx].to_f64().unwrap();
        let temp = (replica_idx as f64 / (self.conf.common.num_replicas - 1) as f64).powf(power);

        let temp_clamped = temp.clamp(1e-10, 1.0 - 1e-10);
        T::from_f64(temp_clamped).unwrap()
    }

    pub fn swap(&mut self) {
        let n = self.population.len();
        if n < 2 {
            return;
        }

        // Try swapping adjacent replicas
        for i in 0..n - 1 {
            let t_i = self.get_temperature(i);
            let t_j = self.get_temperature(i + 1);

            let swap_accepted = self.metropolis_hastings.accept_replica_exchange::<N>(
                &self.fitness[i],
                &self.fitness[i + 1],
                t_i,
                t_j,
            );

            if swap_accepted {
                self.population.swap(i, i + 1);
                self.fitness.swap(i, i + 1);
                self.constraints.swap(i, i + 1);
                self.step_sizes.swap(i, i + 1);
            }

            // Swap acceptance rate with exponential moving average
            let smoothing = self.conf.common.swap_rate_smoothing;
            let current_success = if swap_accepted { 1.0 } else { 0.0 };
            self.swap_acceptance_rates[i] =
                smoothing * current_success + (1.0 - smoothing) * self.swap_acceptance_rates[i];
        }

        if self.conf.common.adaptive_swapping
            && rand::rng().random::<f64>() < self.conf.common.random_swap_probability
        {
            self.attempt_random_swap();
        }
    }

    /// Random non-adjacent swaps
    fn attempt_random_swap(&mut self) {
        let n = self.population.len();
        if n < 3 {
            return;
        }

        let i = rand::rng().random_range(0..n);
        let mut j = rand::rng().random_range(0..n);

        while j == i || j == i.wrapping_sub(1) || j == i + 1 {
            j = rand::rng().random_range(0..n);
        }

        let t_i = self.get_temperature(i);
        let t_j = self.get_temperature(j);

        if self.metropolis_hastings.accept_replica_exchange::<N>(
            &self.fitness[i],
            &self.fitness[j],
            t_i,
            t_j,
        ) {
            self.population.swap(i, j);
            self.fitness.swap(i, j);
            self.constraints.swap(i, j);
            self.step_sizes.swap(i, j);
        }
    }

    pub fn get_replica_population(&self, replica_idx: usize) -> Option<&OMatrix<T, N, D>> {
        self.population.get(replica_idx)
    }

    pub fn get_all_replica_populations(&self) -> &Vec<OMatrix<T, N, D>> {
        &self.population
    }

    pub fn get_num_replicas(&self) -> usize {
        self.population.len()
    }

    /// Set a new preconditioner for covariance matrix computation
    pub fn set_preconditioner(
        &mut self,
        preconditioner: Box<dyn Preconditioner<T, N, D> + Send + Sync>,
    ) {
        self.preconditioner = preconditioner;
        // Recompute covariance matrices with the new preconditioner
        self.update_covariance_matrices();
    }

    pub fn update_covariance_matrices(&mut self) {
        self.covariance_matrices = (0..self.conf.common.num_replicas)
            .into_par_iter()
            .map(|i| {
                self.preconditioner.compute_covariance(
                    &self.population[i],
                    &self.fitness[i],
                    &self.constraints[i],
                )
            })
            .collect();
        self.last_covariance_update = self.st.iter;
    }

    /// Only update cov when acceptance is poor
    fn should_update_covariance(&self) -> bool {
        let iterations_since_update = self.st.iter - self.last_covariance_update;

        let avg_acceptance =
            self.acceptance_rates.iter().sum::<f64>() / self.acceptance_rates.len() as f64;

        let min_freq = self.conf.common.min_covariance_update_freq;
        let base_frequency = if avg_acceptance < 0.2 {
            min_freq // Very low acceptance - barely update
        } else if avg_acceptance < 0.4 {
            min_freq * 2 // Low acceptance - update moderately
        } else {
            min_freq * 4 // Good acceptance - update less frequently
        };

        iterations_since_update >= base_frequency || iterations_since_update >= min_freq * 10
    }

    pub fn get_acceptance_rates(&self) -> &[f64] {
        &self.acceptance_rates
    }

    pub fn get_swap_acceptance_rates(&self) -> &[f64] {
        &self.swap_acceptance_rates
    }

    pub fn compute_effective_sample_size(&self) -> Vec<f64> {
        let mut ess_values = Vec::with_capacity(self.conf.common.num_replicas);

        for replica_idx in 0..self.conf.common.num_replicas {
            let fitness_chain: Vec<f64> = self.fitness[replica_idx]
                .iter()
                .map(|f| f.to_f64().unwrap_or(0.0))
                .collect();

            let ess = self.compute_ess_for_chain(&fitness_chain);
            ess_values.push(ess);
        }

        ess_values
    }

    /// Compute ESS using autocorrelation function
    fn compute_ess_for_chain(&self, chain: &[f64]) -> f64 {
        let n = chain.len();
        if n < 10 {
            return n as f64;
        }

        let mean = chain.iter().sum::<f64>() / n as f64;
        let variance = chain.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

        if variance < 1e-12 {
            return n as f64; // Constant chain, perfect mixing
        }

        let max_lag = (n / 4).min(200);
        let mut autocorr = Vec::with_capacity(max_lag);

        for lag in 0..max_lag {
            let mut sum = 0.0;
            let count = n - lag;

            for i in 0..count {
                sum += (chain[i] - mean) * (chain[i + lag] - mean);
            }

            let rho = sum / (count as f64 * variance);
            autocorr.push(rho);

            if lag > 5 && rho < 0.01 {
                break;
            }
        }

        // Integrated autocorrelation time: τ = 1 + 2 * Σ ρ(k)
        let mut tau_int = 1.0;
        let mut cumsum = 0.0;

        for (k, &rho) in autocorr.iter().enumerate().skip(1) {
            if rho <= 0.0 {
                break;
            }

            cumsum += rho;
            let current_tau = 1.0 + 2.0 * cumsum;

            if k as f64 >= 6.0 * current_tau {
                break;
            }

            tau_int = current_tau;
        }

        let ess = n as f64 / (2.0 * tau_int + 1.0);

        ess.min(n as f64).max(1.0)
    }

    pub fn compute_population_diversity(&self) -> f64 {
        let mut total_variance = 0.0;
        let num_replicas = self.conf.common.num_replicas;

        if num_replicas < 2 {
            return 0.0;
        }

        let replica_best_fitness: Vec<f64> = (0..num_replicas)
            .map(|i| {
                self.fitness[i]
                    .iter()
                    .zip(self.constraints[i].iter())
                    .filter_map(|(f, c)| if *c { Some(f.to_f64().unwrap()) } else { None })
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();

        let mean_fitness = replica_best_fitness.iter().sum::<f64>() / num_replicas as f64;

        for fitness in replica_best_fitness {
            total_variance += (fitness - mean_fitness).powi(2);
        }

        total_variance / num_replicas as f64
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for PT<T, N, D>
where
    T: FloatNum + RealField + Send + Sync + Sum,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync + DimSub<nalgebra::Const<1>>,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N, D>
        + Allocator<N>
        + Allocator<D, D>
        + Allocator<U1, D>
        + Allocator<<D as DimSub<nalgebra::Const<1>>>::Output>,
{
    fn step(&mut self) {
        let temperatures: Vec<T> = (0..self.conf.common.num_replicas)
            .map(|i| self.get_temperature(i))
            .collect();

        let mut new_best_fitness = self.best_fitness;
        let mut new_best_individual = self.best_individual.clone();

        let replica_updates: Vec<_> = (0..self.conf.common.num_replicas)
            .into_par_iter()
            .map(|replica_idx| {
                let temperature = temperatures[replica_idx];
                let alpha = T::from_f64(self.conf.common.alpha).unwrap();
                let omega = T::from_f64(self.conf.common.omega).unwrap();

                let individual_updates: Vec<_> = (0..self.population[replica_idx].nrows())
                    .into_par_iter()
                    .map(|j| {
                        let x_old = self.population[replica_idx].row(j).transpose();
                        // Proposals
                        let x_new = if matches!(self.metropolis_hastings.move_type, crate::algorithms::parallel_tempering::metropolis_hastings::MoveType::PCN) {
                            let variance_param = T::from_f64(1.0).unwrap() / ComplexField::sqrt(T::from_usize(self.st.iter).unwrap() + T::from_f64(1.0).unwrap());
                            self.metropolis_hastings.local_move_pcn_with_variance(
                                &x_old,
                                &self.covariance_matrices[replica_idx],
                                variance_param,
                            )
                        } else if matches!(self.metropolis_hastings.move_type, crate::algorithms::parallel_tempering::metropolis_hastings::MoveType::MALA) && self.metropolis_hastings.mala_use_preconditioning {
                            self.metropolis_hastings.local_move_with_covariance(
                                &x_old,
                                &self.step_sizes[replica_idx][j],
                                &self.covariance_matrices[replica_idx],
                                temperature,
                            )
                        } else {
                            self.metropolis_hastings.local_move(
                                &x_old,
                                &self.step_sizes[replica_idx][j],
                                temperature,
                            )
                        };

                        let fitness_new = self.opt_prob.evaluate(&x_new);
                        let constr_new = self.opt_prob.is_feasible(&x_new);

                        let accepted = self.metropolis_hastings.accept_reject(
                            &x_old,
                            &x_new,
                            constr_new,
                            temperature,
                        );

                        // Step size tuning (Parks et al. 2013)
                        let new_step_size = if accepted {
                            crate::algorithms::parallel_tempering::metropolis_hastings::MetropolisHastings::update_step_size_parks(
                                &self.step_sizes[replica_idx][j],
                                &x_old,
                                &x_new,
                                alpha,
                                omega,
                            )
                        } else {
                            self.step_sizes[replica_idx][j].clone()
                        };

                        (j, x_new, fitness_new, constr_new, accepted, new_step_size)
                    })
                    .collect();

                (replica_idx, individual_updates)
            })
            .collect();

        // Apply updates, not threads safe (race conditions)
        for (replica_idx, individual_updates) in replica_updates {
            let mut accepted_count = 0;
            let total_count = individual_updates.len();

            for (j, x_new, fitness_new, constr_new, accepted, new_step_size) in individual_updates {
                if accepted {
                    accepted_count += 1;
                    self.population[replica_idx]
                        .row_mut(j)
                        .copy_from(&x_new.transpose());
                    self.fitness[replica_idx][j] = fitness_new;
                    self.constraints[replica_idx][j] = constr_new;
                    self.step_sizes[replica_idx][j] = new_step_size;

                    if constr_new && fitness_new > new_best_fitness {
                        new_best_fitness = fitness_new;
                        new_best_individual = x_new;
                    }
                }
            }

            let current_rate = accepted_count as f64 / total_count as f64;
            let smoothing = self.conf.common.acceptance_rate_smoothing;
            self.acceptance_rates[replica_idx] =
                smoothing * current_rate + (1.0 - smoothing) * self.acceptance_rates[replica_idx];
        }

        if new_best_fitness > self.best_fitness {
            self.best_fitness = new_best_fitness;
            self.best_individual = new_best_individual;
        }

        if match &self.swap_check {
            SwapCheck::Periodic(p) => p.should_swap(self.st.iter),
            SwapCheck::Stochastic(s) => s.should_swap(self.st.iter),
            SwapCheck::Always(a) => a.should_swap(self.st.iter),
        } {
            self.swap();
        }

        if self.should_update_covariance() {
            self.update_covariance_matrices();
        }

        let coldest_replica_idx = self.conf.common.num_replicas - 1;
        self.st.best_x = self.best_individual.clone();
        self.st.best_f = self.best_fitness;
        self.st.pop = self.population[coldest_replica_idx].clone();
        self.st.fitness = self.fitness[coldest_replica_idx].clone();
        self.st.constraints = self.constraints[coldest_replica_idx].clone();

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }

    fn get_replica_populations(&self) -> Option<Vec<OMatrix<T, N, D>>> {
        Some(self.population.clone())
    }

    fn get_replica_temperatures(&self) -> Option<Vec<T>> {
        let temperatures: Vec<T> = (0..self.conf.common.num_replicas)
            .map(|i| self.get_temperature(i))
            .collect();
        Some(temperatures)
    }
}

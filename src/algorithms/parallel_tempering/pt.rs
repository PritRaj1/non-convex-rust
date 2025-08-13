use nalgebra::{
    allocator::Allocator, ComplexField, DefaultAllocator, Dim, DimSub, OMatrix, OVector, RealField,
    U1,
};

use std::iter::Sum;

use rayon::prelude::*;

use crate::algorithms::parallel_tempering::{
    metropolis_hastings::MetropolisHastings,
    preconditioners::{Preconditioner, SampleCovariance},
    replica_exchange::{Always, Periodic, Stochastic, SwapCheck},
    replica_state::ReplicaState,
    statistics::Statistics,
    swap_manager::SwapManager,
    temperature::PowerLawScheduler,
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
    pub temperature_manager: PowerLawScheduler<T>,
    pub swap_manager: SwapManager<T, N, D>,
    pub replicas: Vec<ReplicaState<T, N, D>>,
    pub opt_prob: OptProb<T, D>,
    pub best_individual: OVector<T, D>,
    pub best_fitness: T,
    pub preconditioner: Box<dyn Preconditioner<T, N, D> + Send + Sync>,
    pub covariance_matrices: Vec<OMatrix<T, D, D>>,
    pub st: State<T, N, D>,
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

        let temperature_manager = PowerLawScheduler::new(
            conf.common.power_law_init,
            conf.common.power_law_final,
            conf.common.power_law_cycles as f64,
            conf.common.num_replicas,
            max_iter,
        );

        let swap_manager = SwapManager::new(
            metropolis_hastings.clone(),
            conf.common.num_replicas,
            conf.common.adaptive_swapping,
            conf.common.random_swap_probability,
            conf.common.swap_rate_smoothing,
        );

        let step_size_value = match &conf.update_conf {
            UpdateConf::MetropolisHastings(conf) => conf.random_walk_step_size,
            UpdateConf::MALA(conf) => conf.step_size,
            UpdateConf::PCN(conf) => conf.step_size,
            UpdateConf::Auto(_) => conf.common.mala_step_size,
        };

        let mut replicas = Vec::with_capacity(conf.common.num_replicas);
        for _ in 0..conf.common.num_replicas {
            let replica = ReplicaState::new(&init_pop, &opt_prob, step_size_value);
            replicas.push(replica);
        }

        // Find best individual across all replicas
        let mut best_individual = replicas[0].population.row(0).transpose().into_owned();
        let mut best_fitness = replicas[0].fitness[0];

        for replica in &replicas {
            if let Some((individual, fitness)) = replica.find_best_individual() {
                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_individual = individual;
                }
            }
        }

        let preconditioner: Box<dyn Preconditioner<T, N, D> + Send + Sync> =
            Box::new(SampleCovariance::new(T::from_f64(0.01).unwrap()));

        let mut covariance_matrices = Vec::with_capacity(conf.common.num_replicas);
        for replica in &replicas {
            let cov = preconditioner.compute_covariance(
                &replica.population,
                &replica.fitness,
                &replica.constraints,
            );
            covariance_matrices.push(cov);
        }

        let first_replica_pop = replicas[0].population.clone();
        let first_replica_fitness = replicas[0].fitness.clone();
        let first_replica_constraints = replicas[0].constraints.clone();

        Self {
            conf,
            metropolis_hastings,
            swap_check,
            temperature_manager,
            swap_manager,
            replicas,
            opt_prob,
            best_individual: best_individual.clone(),
            best_fitness,
            preconditioner,
            covariance_matrices,
            st: State {
                best_x: best_individual,
                best_f: best_fitness,
                pop: first_replica_pop,
                fitness: first_replica_fitness,
                constraints: first_replica_constraints,
                iter: 1,
            },
            last_covariance_update: 0,
        }
    }

    pub fn swap(&mut self) {
        let temperatures = self.temperature_manager.get_all_temperatures();

        let mut populations: Vec<OMatrix<T, N, D>> =
            self.replicas.iter().map(|r| r.population.clone()).collect();
        let mut fitnesses: Vec<OVector<T, N>> =
            self.replicas.iter().map(|r| r.fitness.clone()).collect();
        let mut constraints: Vec<OVector<bool, N>> = self
            .replicas
            .iter()
            .map(|r| r.constraints.clone())
            .collect();
        let mut step_sizes: Vec<Vec<OMatrix<T, D, D>>> =
            self.replicas.iter().map(|r| r.step_sizes.clone()).collect();

        self.swap_manager.swap_adjacent_replicas(
            &mut populations,
            &mut fitnesses,
            &mut constraints,
            &mut step_sizes,
            &temperatures,
        );

        for (i, replica) in self.replicas.iter_mut().enumerate() {
            replica.population = populations[i].clone();
            replica.fitness = fitnesses[i].clone();
            replica.constraints = constraints[i].clone();
            replica.step_sizes = step_sizes[i].clone();
        }
    }

    pub fn get_replica_population(&self, replica_idx: usize) -> Option<&OMatrix<T, N, D>> {
        self.replicas.get(replica_idx).map(|r| &r.population)
    }

    pub fn get_all_replica_populations(&self) -> Vec<OMatrix<T, N, D>> {
        self.replicas.iter().map(|r| r.population.clone()).collect()
    }

    pub fn get_num_replicas(&self) -> usize {
        self.replicas.len()
    }

    // Preconditioner adaption for MALA and pCN
    pub fn set_preconditioner(
        &mut self,
        preconditioner: Box<dyn Preconditioner<T, N, D> + Send + Sync>,
    ) {
        self.preconditioner = preconditioner;
        self.update_covariance_matrices();
    }

    pub fn update_covariance_matrices(&mut self) {
        self.covariance_matrices = (0..self.conf.common.num_replicas)
            .into_par_iter()
            .map(|i| {
                self.preconditioner.compute_covariance(
                    &self.replicas[i].population,
                    &self.replicas[i].fitness,
                    &self.replicas[i].constraints,
                )
            })
            .collect();
        self.last_covariance_update = self.st.iter;
    }

    /// Only update cov when acceptance is poor
    fn should_update_covariance(&self) -> bool {
        let iterations_since_update = self.st.iter - self.last_covariance_update;

        let avg_acceptance = self.replicas.iter().map(|r| r.acceptance_rate).sum::<f64>()
            / self.replicas.len() as f64;

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

    pub fn get_acceptance_rates(&self) -> Vec<f64> {
        self.replicas.iter().map(|r| r.acceptance_rate).collect()
    }

    pub fn get_swap_acceptance_rates(&self) -> &[f64] {
        self.swap_manager.get_swap_acceptance_rates()
    }

    pub fn compute_effective_sample_size(&self) -> Vec<f64> {
        let fitnesses: Vec<OVector<T, N>> =
            self.replicas.iter().map(|r| r.fitness.clone()).collect();
        Statistics::compute_effective_sample_size(&fitnesses)
    }

    pub fn compute_population_diversity(&self) -> f64 {
        let fitnesses: Vec<OVector<T, N>> =
            self.replicas.iter().map(|r| r.fitness.clone()).collect();
        let constraints: Vec<OVector<bool, N>> = self
            .replicas
            .iter()
            .map(|r| r.constraints.clone())
            .collect();
        Statistics::compute_population_diversity(
            &fitnesses,
            &constraints,
            self.conf.common.num_replicas,
        )
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
        self.temperature_manager.update_iteration(self.st.iter);
        let temperatures = self.temperature_manager.get_all_temperatures();

        let mut new_best_fitness = self.best_fitness;
        let mut new_best_individual = self.best_individual.clone();

        let replica_updates: Vec<_> = (0..self.conf.common.num_replicas)
            .into_par_iter()
            .map(|replica_idx| {
                let temperature = temperatures[replica_idx];
                let alpha = T::from_f64(self.conf.common.alpha).unwrap();
                let omega = T::from_f64(self.conf.common.omega).unwrap();

                let individual_updates: Vec<_> = (0..self.replicas[replica_idx].population.nrows())
                    .into_par_iter()
                    .map(|j| {
                        let x_old = self.replicas[replica_idx].population.row(j).transpose();
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
                                &self.replicas[replica_idx].step_sizes[j],
                                &self.covariance_matrices[replica_idx],
                                temperature,
                            )
                        } else {
                            self.metropolis_hastings.local_move(
                                &x_old,
                                &self.replicas[replica_idx].step_sizes[j],
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
                                &self.replicas[replica_idx].step_sizes[j],
                                &x_old,
                                &x_new,
                                alpha,
                                omega,
                            )
                        } else {
                            self.replicas[replica_idx].step_sizes[j].clone()
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
                    self.replicas[replica_idx]
                        .population
                        .row_mut(j)
                        .copy_from(&x_new.transpose());
                    self.replicas[replica_idx].fitness[j] = fitness_new;
                    self.replicas[replica_idx].constraints[j] = constr_new;
                    self.replicas[replica_idx].step_sizes[j] = new_step_size;

                    if constr_new && fitness_new > new_best_fitness {
                        new_best_fitness = fitness_new;
                        new_best_individual = x_new;
                    }
                }
            }

            let current_rate = accepted_count as f64 / total_count as f64;
            let smoothing = self.conf.common.acceptance_rate_smoothing;
            self.replicas[replica_idx].update_acceptance_rate(current_rate, smoothing);
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
        self.st.pop = self.replicas[coldest_replica_idx].population.clone();
        self.st.fitness = self.replicas[coldest_replica_idx].fitness.clone();
        self.st.constraints = self.replicas[coldest_replica_idx].constraints.clone();

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }

    fn get_replica_populations(&self) -> Option<Vec<OMatrix<T, N, D>>> {
        Some(self.get_all_replica_populations())
    }

    fn get_replica_temperatures(&self) -> Option<Vec<T>> {
        Some(self.temperature_manager.get_all_temperatures())
    }
}

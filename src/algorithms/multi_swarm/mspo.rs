use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rayon::prelude::*;

use crate::algorithms::multi_swarm::information_exchange::InformationExchange;
use crate::algorithms::multi_swarm::population::{
    find_best_solution, get_population, update_population_state,
};
use crate::algorithms::multi_swarm::stagnation_monitor::StagnationMonitor;
use crate::algorithms::multi_swarm::swarm::{initialize_swarms, Swarm};
use crate::utils::config::MSPOConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

pub struct MSPO<T, N, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    N: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D> + Allocator<U1>,
{
    pub conf: MSPOConf,
    pub st: State<T, N, D>,
    pub swarms: Vec<Swarm<T, D>>,
    pub opt_prob: OptProb<T, D>,
    stagnation_monitor: StagnationMonitor<T>,
    information_exchange: InformationExchange<T, D>,
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

        let (best_x, best_fitness) = find_best_solution(&init_pop, &opt_prob);

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

        let improvement_threshold = T::from_f64(conf.improvement_threshold).unwrap();
        let stagnation_monitor = StagnationMonitor::new(improvement_threshold, best_fitness);
        let information_exchange = InformationExchange::new(conf.clone(), opt_prob.clone());

        Self {
            conf,
            st,
            swarms,
            opt_prob,
            stagnation_monitor,
            information_exchange,
        }
    }

    pub fn stagnation_counter(&self) -> usize {
        self.stagnation_monitor.stagnation_counter()
    }

    pub fn is_stagnated(&self) -> bool {
        self.stagnation_monitor.is_stagnated()
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
        self.stagnation_monitor.get_performance_stats()
    }

    pub fn get_population(&self) -> OMatrix<T, N, D> {
        get_population(&self.swarms)
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

        self.stagnation_monitor.check_stagnation(self.st.best_f);

        let exchange_interval = if self.stagnation_monitor.stagnation_counter() > 10 {
            self.conf.exchange_interval / 2 // More frequent exchange when stagnated
        } else {
            self.conf.exchange_interval
        };

        if self.st.iter % exchange_interval == 0 {
            self.information_exchange.exchange_information(
                &mut self.swarms,
                self.stagnation_monitor.stagnation_counter(),
            );
        }

        update_population_state(&mut self.st, &self.swarms, &self.opt_prob);
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

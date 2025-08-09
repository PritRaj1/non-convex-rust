use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rayon::prelude::*;

use crate::utils::alg_conf::sa_conf::SAConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::simulated_annealing::{
    acceptance::MetropolisAcceptance,
    cooling::{CoolingSchedule, ExponentialCooling},
    neighbor_gen::GaussianGenerator,
};

pub struct SimulatedAnnealing<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: SAConf,
    pub opt_prob: OptProb<T, D>,
    pub x: OVector<T, D>,
    pub fitness: T,
    pub constraints: bool,
    pub st: State<T, N, D>,
    pub temperature: T,
    no_improve_count: usize,
    neighbor_gen: GaussianGenerator<T, D>,
    cooling_schedule: ExponentialCooling,
    acceptance: MetropolisAcceptance<T, D>,
}

impl<T, N, D> SimulatedAnnealing<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<N>,
{
    pub fn new(conf: SAConf, init_pop: OMatrix<T, U1, D>, opt_prob: OptProb<T, D>) -> Self {
        let init_x = init_pop.row(0).transpose();
        let best_f = opt_prob.evaluate(&init_x);
        let n = init_x.len();

        Self {
            conf: conf.clone(),
            opt_prob: opt_prob.clone(),
            x: init_x.clone(),
            fitness: best_f,
            constraints: opt_prob.is_feasible(&init_x),
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
            temperature: T::from_f64(conf.initial_temp).unwrap(),
            no_improve_count: 0,
            neighbor_gen: GaussianGenerator::new(
                opt_prob.clone(),
                init_x.clone(),
                T::from_f64(conf.step_size).unwrap(),
            ),
            cooling_schedule: ExponentialCooling,
            acceptance: MetropolisAcceptance::new(opt_prob, init_x),
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for SimulatedAnnealing<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        let min_step = self.conf.step_size * 0.01;
        let step_size = (self.conf.step_size
            * (self.temperature / T::from_f64(self.conf.initial_temp).unwrap())
                .to_f64()
                .unwrap()
                .sqrt())
        .max(min_step);

        let bounds = (
            T::from_f64(self.conf.x_min).unwrap(),
            T::from_f64(self.conf.x_max).unwrap(),
        );

        let neighbors: Vec<_> = (0..self.conf.num_neighbors)
            .into_par_iter()
            .map(|_| {
                self.neighbor_gen
                    .generate(&self.x, step_size, bounds, self.temperature)
            })
            .collect();

        let mut improved = false;
        for neighbor in neighbors {
            let neighbor_fitness = self.opt_prob.evaluate(&neighbor);

            if neighbor_fitness > self.st.best_f && self.opt_prob.is_feasible(&neighbor) {
                self.st.best_f = neighbor_fitness;
                self.st.best_x = neighbor.clone();
                self.no_improve_count = 0;
                improved = true;
            }

            let feasible = self.opt_prob.is_feasible(&neighbor);

            // Use Metropolis criterion for current solution
            if self.acceptance.accept(
                &self.x,
                self.fitness,
                &neighbor,
                neighbor_fitness,
                self.temperature,
                T::from_f64(step_size).unwrap(),
            ) && feasible
            {
                self.x = neighbor;
                self.fitness = neighbor_fitness;
                self.constraints = feasible;
            }
        }

        if !improved {
            self.no_improve_count += 1;
        }

        // More aggressive reheating if stuck for a long time
        if self.no_improve_count > self.conf.reheat_after * 2 {
            self.temperature = self
                .cooling_schedule
                .reheat(T::from_f64(self.conf.initial_temp).unwrap());
            self.no_improve_count = 0;
            self.x = self.st.best_x.clone(); // Reset to best known solution
        }

        if self.fitness > self.st.best_f {
            self.st.best_f = self.fitness;
            self.st.best_x = self.x.clone();
        }

        self.st.pop.row_mut(0).copy_from(&self.x.transpose());
        self.st.fitness[0] = self.fitness;
        self.st.constraints[0] = self.constraints;

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

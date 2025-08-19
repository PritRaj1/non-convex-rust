use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};

pub mod algorithms;
pub mod utils;
use crate::utils::config::{AlgConf, Config, OptConf};

use crate::algorithms::{
    adam::adam_opt::Adam, cma_es::cma_es_opt::CMAES, continous_ga::cga::CGA,
    differential_evolution::de::DE, grasp::grasp_opt::GRASP, limited_memory_bfgs::lbfgs::LBFGS,
    multi_swarm::mspo::MSPO, nelder_mead::nm::NelderMead, parallel_tempering::pt::PT,
    sg_ascent::sga::SGAscent, simulated_annealing::sa::SimulatedAnnealing,
    tabu_search::tabu::TabuSearch,
};

use crate::utils::opt_prob::{
    BooleanConstraintFunction, FloatNumber as FloatNum, ObjectiveFunction, OptProb,
    OptimizationAlgorithm, State,
};

pub struct Result<T, N, D>
where
    T: FloatNum,
    D: Dim,
    N: Dim,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub best_x: OVector<T, D>,
    pub best_f: T,
    pub final_pop: OMatrix<T, N, D>,
    pub final_fitness: OVector<T, N>,
    pub final_constraints: OVector<bool, N>,
    pub convergence_iter: usize,
}

pub struct NonConvexOpt<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<bool, N>: Send + Sync,
    OVector<bool, D>: Send + Sync,
    OMatrix<bool, U1, N>: Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, U1, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N>
        + Allocator<N, D>
        + Allocator<D, D>
        + Allocator<U1, D>
        + Allocator<U1, N>,
{
    pub alg: Box<dyn OptimizationAlgorithm<T, N, D>>,
    pub conf: OptConf,
    pub converged: bool,
    best_fitness_history: Vec<T>,
}

impl<T, N, D> NonConvexOpt<T, N, D>
where
    T: FloatNum + nalgebra::RealField + std::iter::Sum,
    N: Dim,
    D: Dim + nalgebra::DimSub<nalgebra::Const<1>>,
    OVector<bool, N>: Send + Sync,
    OVector<bool, D>: Send + Sync,
    OMatrix<bool, U1, N>: Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    OMatrix<T, U1, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
        + Allocator<N>
        + Allocator<N, D>
        + Allocator<D, D>
        + Allocator<U1, D>
        + Allocator<U1, N>
        + Allocator<<D as nalgebra::DimSub<nalgebra::Const<1>>>::Output>,
{
    pub fn new<
        F: ObjectiveFunction<T, D> + 'static,
        G: BooleanConstraintFunction<T, D> + 'static,
    >(
        conf: Config,
        init_pop: OMatrix<T, N, D>,
        obj_f: F,
        constr_f: Option<G>,
    ) -> Self {
        let opt_prob = OptProb::new(
            Box::new(obj_f),
            match constr_f {
                Some(constr_f) => Some(Box::new(constr_f)),
                None => None,
            },
        );

        let alg: Box<dyn OptimizationAlgorithm<T, N, D>> = match conf.alg_conf {
            AlgConf::CGA(cga_conf) => Box::new(CGA::new(
                cga_conf,
                init_pop,
                opt_prob,
                conf.opt_conf.max_iter,
            )),
            AlgConf::PT(pt_conf) => {
                Box::new(PT::new(pt_conf, init_pop, opt_prob, conf.opt_conf.max_iter))
            }
            AlgConf::TS(ts_conf) => Box::new(TabuSearch::new(
                ts_conf,
                init_pop.row(0).into_owned(),
                opt_prob,
            )),
            AlgConf::Adam(adam_conf) => {
                Box::new(Adam::new(adam_conf, init_pop.row(0).into_owned(), opt_prob))
            }
            AlgConf::GRASP(grasp_conf) => Box::new(GRASP::new(
                grasp_conf,
                init_pop.row(0).into_owned(),
                opt_prob,
            )),
            AlgConf::SGA(sga_conf) => Box::new(SGAscent::new(
                sga_conf,
                init_pop.row(0).into_owned(),
                opt_prob,
            )),
            AlgConf::NM(nm_conf) => Box::new(NelderMead::new(nm_conf, init_pop, opt_prob)),
            AlgConf::LBFGS(lbfgs_conf) => Box::new(LBFGS::new(
                lbfgs_conf,
                init_pop.row(0).into_owned(),
                opt_prob,
            )),
            AlgConf::MSPO(mspo_conf) => Box::new(MSPO::new(
                mspo_conf,
                init_pop,
                opt_prob,
                conf.opt_conf.max_iter,
            )),
            AlgConf::SA(sa_conf) => Box::new(SimulatedAnnealing::new(
                sa_conf,
                init_pop.row(0).into_owned(),
                opt_prob,
            )),
            AlgConf::DE(de_conf) => Box::new(DE::new(de_conf, init_pop, opt_prob)),
            AlgConf::CMAES(cma_es_conf) => Box::new(CMAES::new(cma_es_conf, init_pop, opt_prob)),
        };

        Self {
            alg,
            conf: conf.opt_conf,
            converged: false,
            best_fitness_history: Vec::new(),
        }
    }

    fn check_convergence(&self, current_best: T, previous_best: T) -> bool {
        let atol = T::from_f64(self.conf.atol).unwrap();
        let rtol = T::from_f64(self.conf.rtol).unwrap();
        let min_iter_for_rtol =
            (self.conf.max_iter as f64 * self.conf.rtol_max_iter_fraction).floor() as usize;

        let improvement = current_best - previous_best;
        let abs_improvement = num_traits::Float::abs(improvement);

        let abs_converged = abs_improvement < atol && self.alg.state().iter > min_iter_for_rtol;

        let rel_converged = if num_traits::Float::abs(current_best) > T::from_f64(1e-10).unwrap() {
            abs_improvement / num_traits::Float::abs(current_best) <= rtol
        } else {
            abs_improvement <= atol
        };

        // Check for stagnation: no significant improvement over a window of iterations
        let stagnation_converged = if self.best_fitness_history.len() >= self.conf.stagnation_window
            && self.alg.state().iter > min_iter_for_rtol
        {
            let window_start = self.best_fitness_history.len() - self.conf.stagnation_window;
            let oldest_in_window = self.best_fitness_history[window_start];
            let stagnation_improvement = current_best - oldest_in_window;
            let abs_stagnation_improvement = num_traits::Float::abs(stagnation_improvement);

            abs_stagnation_improvement < atol
                || (num_traits::Float::abs(current_best) > T::from_f64(1e-10).unwrap()
                    && abs_stagnation_improvement / num_traits::Float::abs(current_best) <= rtol)
        } else {
            false
        };

        let converged = abs_converged
            || (rel_converged && self.alg.state().iter > min_iter_for_rtol)
            || stagnation_converged;

        if converged {
            let reason = if abs_converged {
                "absolute tolerance"
            } else if rel_converged && self.alg.state().iter > min_iter_for_rtol {
                "relative tolerance"
            } else if stagnation_converged {
                "stagnation"
            } else {
                "unknown"
            };

            println!(
                "Converged in {} iterations due to {} (improvement: {:.2e})",
                self.alg.state().iter,
                reason,
                improvement.to_f64().unwrap_or(0.0)
            );
        }

        converged
    }

    pub fn step(&mut self) {
        if self.converged {
            return;
        }

        let previous_best_fitness = self.alg.state().best_f;
        self.alg.step();
        let current_best_fitness = self.alg.state().best_f;

        // Track fitness history for stagnation detection
        self.best_fitness_history.push(current_best_fitness);

        // Keep history bounded to avoid unbounded memory growth
        let max_history = self.conf.stagnation_window * 2;
        if self.best_fitness_history.len() > max_history {
            let excess = self.best_fitness_history.len() - max_history;
            self.best_fitness_history.drain(0..excess);
        }

        self.converged = self.check_convergence(current_best_fitness, previous_best_fitness);
    }

    pub fn run(&mut self) -> &State<T, N, D> {
        while !self.converged && self.alg.state().iter < self.conf.max_iter {
            self.step();
        }
        self.alg.state()
    }

    pub fn get_best_individual(&self) -> OVector<T, D> {
        self.alg.state().best_x.clone()
    }

    pub fn get_population(&self) -> OMatrix<T, N, D> {
        self.alg.state().pop.clone()
    }

    pub fn get_pt_replica_populations(&self) -> Option<Vec<OMatrix<T, N, D>>> {
        self.alg.get_replica_populations()
    }

    pub fn get_pt_replica_temperatures(&self) -> Option<Vec<T>> {
        self.alg.get_replica_temperatures()
    }
}

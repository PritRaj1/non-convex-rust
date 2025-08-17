use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use std::collections::VecDeque;

use crate::utils::config::{LBFGSConf, LineSearchConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::limited_memory_bfgs::linesearch::{
    BacktrackingLineSearch, GoldenSectionLineSearch, HagerZhangLineSearch, LineSearch,
    MoreThuenteLineSearch, StrongWolfeLineSearch,
};

pub struct LBFGS<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: LBFGSConf,
    pub opt_prob: OptProb<T, D>,
    pub x: OVector<T, D>,
    pub st: State<T, N, D>,
    pub linesearch: Box<dyn LineSearch<T, D> + Send + Sync>,

    // L-BFGS vectors
    s: Vec<OVector<T, D>>,
    y: Vec<OVector<T, D>>,

    // Bounds handling
    has_bounds: bool,
    lower_bounds: Option<OVector<T, D>>,
    upper_bounds: Option<OVector<T, D>>,

    // Adaptive parameters
    current_memory_size: usize,
    current_scaling_factor: T,

    // Stagnation
    stagnation_counter: usize,
    restart_counter: usize,
    last_restart_iter: usize,
    last_improvement: T,
    success_history: VecDeque<bool>,
    improvement_history: VecDeque<f64>,
    gradient_history: VecDeque<f64>,
    function_evaluations: usize,
    gradient_evaluations: usize,
}

impl<T, N, D> LBFGS<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<N>,
{
    pub fn new(conf: LBFGSConf, init_pop: OMatrix<T, U1, D>, opt_prob: OptProb<T, D>) -> Self {
        let init_x = init_pop.row(0).transpose();
        let best_f = opt_prob.evaluate(&init_x);
        let n = init_x.len();

        let linesearch: Box<dyn LineSearch<T, D> + Send + Sync> = match &conf.line_search {
            LineSearchConf::Backtracking(backtracking_conf) => {
                Box::new(BacktrackingLineSearch::new(backtracking_conf))
            }
            LineSearchConf::StrongWolfe(strong_wolfe_conf) => {
                Box::new(StrongWolfeLineSearch::new(strong_wolfe_conf))
            }
            LineSearchConf::HagerZhang(hager_zhang_conf) => {
                Box::new(HagerZhangLineSearch::new(hager_zhang_conf))
            }
            LineSearchConf::MoreThuente(more_thuente_conf) => {
                Box::new(MoreThuenteLineSearch::new(more_thuente_conf))
            }
            LineSearchConf::GoldenSection(golden_section_conf) => {
                Box::new(GoldenSectionLineSearch::new(golden_section_conf))
            }
        };

        // Check if problem has bounds
        let lower_bounds = opt_prob.objective.x_lower_bound(&init_x);
        let upper_bounds = opt_prob.objective.x_upper_bound(&init_x);
        let has_bounds = lower_bounds.is_some() || upper_bounds.is_some();
        let current_memory_size = conf.common.memory_size;
        let current_scaling_factor =
            T::from_f64(conf.advanced.numerical_safeguards.scaling_factor).unwrap();

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
            linesearch,
            s: Vec::with_capacity(current_memory_size),
            y: Vec::with_capacity(current_memory_size),
            has_bounds,
            lower_bounds,
            upper_bounds,
            current_memory_size,
            current_scaling_factor,
            stagnation_counter: 0,
            restart_counter: 0,
            last_restart_iter: 0,
            last_improvement: best_f,
            success_history: VecDeque::with_capacity(conf.advanced.success_history_size),
            improvement_history: VecDeque::with_capacity(conf.advanced.improvement_history_size),
            gradient_history: VecDeque::with_capacity(conf.advanced.improvement_history_size),
            function_evaluations: 1,
            gradient_evaluations: 0,
        }
    }

    fn project_onto_bounds(&self, x: &mut OVector<T, D>) {
        if let Some(ref lb) = self.lower_bounds {
            for i in 0..x.len() {
                x[i] = x[i].max(lb[i]);
            }
        }
        if let Some(ref ub) = self.upper_bounds {
            for i in 0..x.len() {
                x[i] = x[i].min(ub[i]);
            }
        }
    }

    fn compute_cauchy_point(&self, g: &OVector<T, D>) -> OVector<T, D> {
        let mut t = T::one();
        let mut x_cp = self.x.clone();

        for i in 0..g.len() {
            if g[i] != T::zero() {
                if let (Some(ref lb), Some(ref ub)) = (&self.lower_bounds, &self.upper_bounds) {
                    if g[i] < T::zero() {
                        t = t.min((ub[i] - self.x[i]) / g[i]);
                    } else {
                        t = t.min((lb[i] - self.x[i]) / g[i]);
                    }
                }
            }
        }

        x_cp -= g * t;
        self.project_onto_bounds(&mut x_cp);
        x_cp
    }

    fn step_with_bounds(&mut self, g: &OVector<T, D>) {
        let x_cp = self.compute_cauchy_point(g);
        let mut p = x_cp - &self.st.best_x;

        let r = self.compute_reduced_gradient(g);
        let z = self.two_loop_recursion(&r);

        self.update_search_direction(&mut p, &z);

        let alpha = self
            .linesearch
            .search(&self.st.best_x, &p, self.st.best_f, g, &self.opt_prob);
        let mut x_new = &self.x + &p * alpha;
        self.project_onto_bounds(&mut x_new);

        self.update_s_y_vectors(&x_new, g);
        self.update_best_solution(&x_new);
        self.x = x_new;
    }

    fn step_without_bounds(&mut self, g: &OVector<T, D>) {
        let p = self.compute_search_direction(g);

        let alpha = self
            .linesearch
            .search(&self.st.best_x, &p, self.st.best_f, g, &self.opt_prob);

        let x_new = &self.x + &p * alpha;

        self.update_s_y_vectors(&x_new, g);
        self.update_best_solution(&x_new);
        self.x = x_new;
    }

    fn compute_reduced_gradient(&self, g: &OVector<T, D>) -> OVector<T, D> {
        let mut r = OVector::<T, D>::zeros_generic(D::from_usize(self.st.best_x.len()), U1);
        for i in 0..self.st.best_x.len() {
            if !self.is_at_bound(i) {
                r[i] = g[i];
            }
        }
        r
    }

    // Two-loop recursion to approximate the inverse Hessian
    fn two_loop_recursion(&self, r: &OVector<T, D>) -> OVector<T, D> {
        let m = self.current_memory_size;
        let mut q = r.clone();
        let mut alpha = vec![T::zero(); m];
        let mut rho = vec![T::zero(); m];

        // First loop: backward
        for i in (0..self.s.len()).rev() {
            let s_dot_y = self.s[i].dot(&self.y[i]);

            // Check conditioning
            if s_dot_y.abs()
                < T::from_f64(
                    self.conf
                        .advanced
                        .numerical_safeguards
                        .conditioning_threshold,
                )
                .unwrap()
            {
                continue; // Skip ill-conditioned pairs
            }

            rho[i] = T::one() / s_dot_y;
            alpha[i] = rho[i] * self.s[i].dot(&q);
            q -= &self.y[i] * alpha[i];
        }

        // Scale
        let mut z = if self.conf.advanced.numerical_safeguards.use_scaling && !self.s.is_empty() {
            let gamma = self.s.last().unwrap().dot(self.y.last().unwrap())
                / self.y.last().unwrap().dot(self.y.last().unwrap());
            q * gamma
        } else {
            q.clone()
        };

        // Second loop: forward
        for i in 0..self.s.len() {
            if rho[i] != T::zero() {
                let beta = rho[i] * self.y[i].dot(&z);
                z += &self.s[i] * (alpha[i] - beta);
            }
        }
        z
    }

    fn update_search_direction(&self, p: &mut OVector<T, D>, z: &OVector<T, D>) {
        for i in 0..self.st.best_x.len() {
            if !self.is_at_bound(i) {
                p[i] = z[i];
            }
        }
    }

    fn compute_search_direction(&self, g: &OVector<T, D>) -> OVector<T, D> {
        let m = self.current_memory_size;
        let mut q = g.clone();
        let mut alpha = vec![T::zero(); m];
        let mut rho = vec![T::zero(); m];

        // First loop: backward
        for i in (0..m).rev() {
            if i < self.s.len() {
                let s_dot_y = self.y[i].dot(&self.s[i]);

                // Check conditioning
                if s_dot_y.abs()
                    < T::from_f64(
                        self.conf
                            .advanced
                            .numerical_safeguards
                            .conditioning_threshold,
                    )
                    .unwrap()
                {
                    continue;
                }

                rho[i] = T::one() / s_dot_y;
                alpha[i] = rho[i] * self.s[i].dot(&q);
                q -= &self.y[i] * alpha[i];
            }
        }

        // Scale
        if !self.s.is_empty() {
            let gamma = self.s.last().unwrap().dot(self.y.last().unwrap())
                / self.y.last().unwrap().dot(self.y.last().unwrap());
            q *= gamma;
        }

        // Second loop: forward
        let mut p = q.clone();
        for i in 0..m {
            if i < self.s.len() && rho[i] != T::zero() {
                let beta = rho[i] * self.y[i].dot(&p);
                p += &self.s[i] * (alpha[i] - beta);
            }
        }
        p
    }

    fn update_s_y_vectors(&mut self, x_new: &OVector<T, D>, g: &OVector<T, D>) {
        let s_new = x_new - &self.st.best_x;
        let y_new = self.opt_prob.objective.gradient(x_new).unwrap() - g;
        self.gradient_evaluations += 1;

        // Check curvature
        let curvature = s_new.dot(&y_new);
        if curvature
            > T::from_f64(self.conf.advanced.numerical_safeguards.curvature_threshold).unwrap()
        {
            if self.s.len() == self.current_memory_size {
                self.s.remove(0);
                self.y.remove(0);
            }
            self.s.push(s_new);
            self.y.push(y_new);
        }
    }

    fn update_best_solution(&mut self, x_new: &OVector<T, D>) {
        let f_new = self.opt_prob.evaluate(x_new);
        self.function_evaluations += 1;

        if f_new > self.st.best_f {
            let improvement = f_new - self.st.best_f;
            self.last_improvement = f_new;
            self.st.best_f = f_new;
            self.st.best_x = x_new.clone();

            self.success_history.push_back(true);
            self.improvement_history
                .push_back(improvement.to_f64().unwrap_or(0.0));

            self.stagnation_counter = 0;
        } else {
            self.success_history.push_back(false);
            self.improvement_history.push_back(0.0);
            self.stagnation_counter += 1;
        }

        if self.success_history.len() > self.conf.advanced.success_history_size {
            self.success_history.pop_front();
        }
        if self.improvement_history.len() > self.conf.advanced.improvement_history_size {
            self.improvement_history.pop_front();
        }
    }

    fn is_at_bound(&self, i: usize) -> bool {
        let at_lower = self
            .lower_bounds
            .as_ref()
            .is_some_and(|lb| self.st.best_x[i] == lb[i]);
        let at_upper = self
            .upper_bounds
            .as_ref()
            .is_some_and(|ub| self.st.best_x[i] == ub[i]);
        at_lower || at_upper
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

        let adaptation_rate = T::from_f64(self.conf.advanced.adaptation_rate).unwrap();

        if self.conf.advanced.memory_adaptation.adaptive_memory {
            if success_rate < 0.2 {
                self.current_memory_size = (self.current_memory_size + 1)
                    .min(self.conf.advanced.memory_adaptation.max_memory_size);
            } else if success_rate > 0.6 {
                self.current_memory_size = (self.current_memory_size.saturating_sub(1))
                    .max(self.conf.advanced.memory_adaptation.min_memory_size); // Decrease memory
            }
        }

        if self.conf.advanced.numerical_safeguards.use_scaling {
            if success_rate < 0.2 {
                self.current_scaling_factor *= T::one() + adaptation_rate;
            } else if success_rate > 0.6 {
                self.current_scaling_factor *=
                    T::one() - adaptation_rate * T::from_f64(0.3).unwrap();
            }
        }
    }

    fn check_stagnation(&self) -> bool {
        let stagnation_window = self.conf.advanced.stagnation_detection.stagnation_window;
        let improvement_threshold = self
            .conf
            .advanced
            .stagnation_detection
            .improvement_threshold;
        let gradient_threshold = self.conf.advanced.stagnation_detection.gradient_threshold;

        if self.improvement_history.len() >= stagnation_window {
            let recent_improvements: Vec<f64> = self
                .improvement_history
                .iter()
                .rev()
                .take(stagnation_window)
                .cloned()
                .collect();

            let avg_improvement =
                recent_improvements.iter().sum::<f64>() / recent_improvements.len() as f64;
            if avg_improvement < improvement_threshold {
                return true;
            }
        }

        if self.gradient_history.len() >= stagnation_window {
            let recent_gradients: Vec<f64> = self
                .gradient_history
                .iter()
                .rev()
                .take(stagnation_window)
                .cloned()
                .collect();

            let avg_gradient = recent_gradients.iter().sum::<f64>() / recent_gradients.len() as f64;
            if avg_gradient < gradient_threshold {
                return true;
            }
        }

        false
    }

    fn check_restart(&mut self) -> bool {
        match &self.conf.advanced.restart_strategy {
            crate::utils::alg_conf::lbfgs_conf::RestartStrategy::None => false,
            crate::utils::alg_conf::lbfgs_conf::RestartStrategy::Periodic { frequency } => {
                self.st.iter - self.last_restart_iter >= *frequency
            }
            crate::utils::alg_conf::lbfgs_conf::RestartStrategy::Stagnation {
                max_iterations,
                threshold,
            } => {
                self.stagnation_counter >= *max_iterations
                    || self.last_improvement.to_f64().unwrap_or(0.0) < *threshold
            }
            crate::utils::alg_conf::lbfgs_conf::RestartStrategy::Adaptive {
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
        let _current_best = self.st.best_x.clone();
        let current_best_f = self.st.best_f;

        self.s.clear();
        self.y.clear();

        self.current_memory_size = self.conf.common.memory_size;
        self.current_scaling_factor =
            T::from_f64(self.conf.advanced.numerical_safeguards.scaling_factor).unwrap();

        self.stagnation_counter = 0;
        self.last_improvement = current_best_f;
        self.restart_counter += 1;
        self.last_restart_iter = self.st.iter;

        self.success_history.clear();
        self.improvement_history.clear();
        self.gradient_history.clear();
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for LBFGS<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        if self.check_restart() {
            self.perform_restart();
        }

        if self.check_stagnation() {
            self.stagnation_counter += 1;
        }

        let g = self.opt_prob.objective.gradient(&self.x).unwrap();
        self.gradient_evaluations += 1;

        let gradient_norm = g.dot(&g).sqrt();
        self.gradient_history
            .push_back(gradient_norm.to_f64().unwrap_or(0.0));
        if self.gradient_history.len() > self.conf.advanced.improvement_history_size {
            self.gradient_history.pop_front();
        }

        if self.has_bounds {
            self.step_with_bounds(&g); // L-BFGS-B
        } else {
            self.step_without_bounds(&g); // L-BFGS
        }

        let fitness = self.opt_prob.evaluate(&self.x);
        self.function_evaluations += 1;
        let constraints = self.opt_prob.is_feasible(&self.x);

        if fitness > self.st.best_f {
            self.st.best_f = fitness;
            self.st.best_x = self.x.clone();
        }

        self.st.pop.row_mut(0).copy_from(&self.x.transpose());
        self.st.fitness[0] = fitness;
        self.st.constraints[0] = constraints;

        self.adapt_parameters();

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

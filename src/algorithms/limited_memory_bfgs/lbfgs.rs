use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};

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
    s: Vec<OVector<T, D>>,
    y: Vec<OVector<T, D>>,
    has_bounds: bool,
    lower_bounds: Option<OVector<T, D>>,
    upper_bounds: Option<OVector<T, D>>,
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

        Self {
            conf,
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
            s: Vec::new(),
            y: Vec::new(),
            has_bounds,
            lower_bounds,
            upper_bounds,
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
        let m = self.conf.common.memory_size;
        let mut q = r.clone();
        let mut alpha = vec![T::zero(); m];
        let mut rho = vec![T::zero(); m];

        for i in (0..self.s.len()).rev() {
            rho[i] = T::one() / self.s[i].dot(&self.y[i]);
            alpha[i] = rho[i] * self.s[i].dot(&q);
            q -= &self.y[i] * alpha[i];
        }

        let mut z = q.clone();
        for i in 0..self.s.len() {
            let beta = rho[i] * self.y[i].dot(&z);
            z += &self.s[i] * (alpha[i] - beta);
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
        let m = self.conf.common.memory_size;
        let mut q = g.clone();
        let mut alpha = vec![T::zero(); m];
        let mut rho = vec![T::zero(); m];

        for i in (0..m).rev() {
            if i < self.s.len() {
                rho[i] = T::one() / self.y[i].dot(&self.s[i]);
                alpha[i] = rho[i] * self.s[i].dot(&q);
                q -= &self.y[i] * alpha[i];
            }
        }

        if !self.s.is_empty() {
            let gamma = self.s.last().unwrap().dot(self.y.last().unwrap())
                / self.y.last().unwrap().dot(self.y.last().unwrap());
            q *= gamma;
        }

        let mut p = q.clone();
        for i in 0..m {
            if i < self.s.len() {
                let beta = rho[i] * self.y[i].dot(&p);
                p += &self.s[i] * (alpha[i] - beta);
            }
        }
        p
    }

    fn update_s_y_vectors(&mut self, x_new: &OVector<T, D>, g: &OVector<T, D>) {
        let s_new = x_new - &self.st.best_x;
        let y_new = self.opt_prob.objective.gradient(x_new).unwrap() - g;

        if self.s.len() == self.conf.common.memory_size {
            self.s.remove(0);
            self.y.remove(0);
        }
        self.s.push(s_new);
        self.y.push(y_new);
    }

    fn update_best_solution(&mut self, x_new: &OVector<T, D>) {
        let f_new = self.opt_prob.evaluate(x_new);
        if f_new > self.st.best_f {
            self.st.best_f = f_new;
            self.st.best_x = x_new.clone();
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
        let g = self.opt_prob.objective.gradient(&self.x).unwrap();

        if self.has_bounds {
            self.step_with_bounds(&g); // L-BFGS-B
        } else {
            self.step_without_bounds(&g); // L-BFGS
        }

        let fitness = self.opt_prob.evaluate(&self.x);
        let constraints = self.opt_prob.is_feasible(&self.x);

        if fitness > self.st.best_f {
            self.st.best_f = fitness;
            self.st.best_x = self.x.clone();
        }

        self.st.pop.row_mut(0).copy_from(&self.x.transpose());
        self.st.fitness[0] = fitness;
        self.st.constraints[0] = constraints;

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

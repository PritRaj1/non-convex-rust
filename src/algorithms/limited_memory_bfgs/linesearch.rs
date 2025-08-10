use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};

use crate::utils::config::{
    BacktrackingConf, GoldenSectionConf, HagerZhangConf, MoreThuenteConf, StrongWolfeConf,
};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

pub trait LineSearch<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        f: T,
        g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T;
}

pub struct BacktrackingLineSearch {
    conf: BacktrackingConf,
}

impl BacktrackingLineSearch {
    pub fn new(conf: &BacktrackingConf) -> Self {
        Self { conf: conf.clone() }
    }
}

impl<T, D> LineSearch<T, D> for BacktrackingLineSearch
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        f: T,
        g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T {
        let mut alpha = T::one();
        let mut x_new = x + p * alpha;

        // Repeat until the Armijo condition is satisfied (for maximization)
        while opt_prob.evaluate(&x_new) < f + T::from_f64(self.conf.c1).unwrap() * alpha * g.dot(p)
        {
            alpha *= T::from_f64(self.conf.rho).unwrap();
            x_new = x + p * alpha;
        }

        alpha
    }
}

pub struct StrongWolfeLineSearch {
    conf: StrongWolfeConf,
}

impl StrongWolfeLineSearch {
    pub fn new(conf: &StrongWolfeConf) -> Self {
        Self { conf: conf.clone() }
    }
}

impl<T, D> LineSearch<T, D> for StrongWolfeLineSearch
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        f: T,
        g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T {
        let c1 = T::from_f64(self.conf.c1).unwrap();
        let c2 = T::from_f64(self.conf.c2).unwrap();
        let mut alpha = T::one();
        let mut alpha_low = T::zero();
        let mut alpha_high = T::from_f64(10.0).unwrap();
        let initial_gp = g.dot(p);

        for _ in 0..self.conf.max_iters {
            let x_new = x + p * alpha;
            let f_new = opt_prob.evaluate(&x_new);
            let g_new = opt_prob.objective.gradient(&x_new).unwrap();
            let g_new_p = g_new.dot(p);

            // For maximization:
            if f_new < f + c1 * alpha * initial_gp {
                // Armijo condition
                alpha_high = alpha;
            } else if g_new_p.abs() < -c2 * initial_gp {
                // Wolfe condition for maximization
                alpha_low = alpha;
            } else {
                return alpha; // Both conditions satisfied
            }

            if alpha_high < T::from_f64(10.0).unwrap() {
                alpha = (alpha_low + alpha_high) / T::from_f64(2.0).unwrap();
            } else {
                alpha *= T::from_f64(2.0).unwrap();
            }
        }
        alpha
    }
}

pub struct HagerZhangLineSearch {
    conf: HagerZhangConf,
}

impl HagerZhangLineSearch {
    pub fn new(conf: &HagerZhangConf) -> Self {
        Self { conf: conf.clone() }
    }
}

impl<T, D> LineSearch<T, D> for HagerZhangLineSearch
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        f: T,
        g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T {
        let c1 = T::from_f64(self.conf.c1).unwrap();
        let c2 = T::from_f64(self.conf.c2).unwrap();
        let theta = T::from_f64(self.conf.theta).unwrap();
        let gamma = T::from_f64(self.conf.gamma).unwrap();
        let mut alpha = T::one();
        let initial_gp = g.dot(p);

        for _ in 0..self.conf.max_iters {
            let x_new = x + p * alpha;
            let f_new = opt_prob.evaluate(&x_new);
            let g_new = opt_prob.objective.gradient(&x_new).unwrap();
            let g_new_p = g_new.dot(p);

            // For maximization:
            if f_new < f + c1 * alpha * initial_gp {
                alpha *= gamma; // Reduce step size
                continue;
            }

            if g_new_p < -c2 * initial_gp {
                // Modified for maximization
                let delta =
                    theta * alpha * (initial_gp - g_new_p) / (f_new - f - alpha * initial_gp);
                alpha += delta;
                continue;
            }

            return alpha;
        }
        alpha
    }
}

pub struct MoreThuenteLineSearch {
    conf: MoreThuenteConf,
}

impl MoreThuenteLineSearch {
    pub fn new(conf: &MoreThuenteConf) -> Self {
        Self { conf: conf.clone() }
    }
}

impl<T, D> LineSearch<T, D> for MoreThuenteLineSearch
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        f: T,
        g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T {
        let ftol = T::from_f64(self.conf.ftol).unwrap();
        let gtol = T::from_f64(self.conf.gtol).unwrap();

        let mut alpha = T::one();
        let mut alpha_low = T::zero();
        let mut alpha_high = T::from_f64(10.0).unwrap();
        let initial_gp = g.dot(p);

        for _ in 0..self.conf.max_iters {
            let x_new = x + p * alpha;
            let f_new = opt_prob.evaluate(&x_new);
            let g_new = opt_prob.objective.gradient(&x_new).unwrap();
            let g_new_p = g_new.dot(p);

            // For maximization:
            if f_new < f + ftol * alpha * initial_gp {
                alpha_high = alpha;
            } else if g_new_p < -gtol * initial_gp {
                alpha_low = alpha;
            } else {
                return alpha;
            }

            alpha = (alpha_low + alpha_high) / T::from_f64(2.0).unwrap();
        }
        alpha
    }
}

pub struct GoldenSectionLineSearch {
    conf: GoldenSectionConf,
}

impl GoldenSectionLineSearch {
    pub fn new(conf: &GoldenSectionConf) -> Self {
        Self { conf: conf.clone() }
    }

    // Helper function to bracket the maximum
    fn bracket_maximum<T: FloatNum, D: Dim>(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> (T, T, T)
    where
        DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
    {
        let golden_ratio: T = T::from_f64((5.0_f64).sqrt() * 0.5 + 0.5).unwrap();
        let bracket_factor = T::from_f64(self.conf.bracket_factor).unwrap();

        let mut a = T::zero();
        let mut b = T::one();
        let mut c = b * golden_ratio;

        let mut fa = opt_prob.evaluate(&(x + p * a));
        let mut fb = opt_prob.evaluate(&(x + p * b));
        let mut fc = opt_prob.evaluate(&(x + p * c));

        // Expand the bracket until we find a triplet where the middle point is higher
        while fb < fa || fb < fc {
            if fb < fa {
                c = b;
                b = a;
                a = b / bracket_factor;
                fc = fb;
                fb = fa;
                fa = opt_prob.evaluate(&(x + p * a));
            } else {
                a = b;
                b = c;
                c = b * bracket_factor;
                fa = fb;
                fb = fc;
                fc = opt_prob.evaluate(&(x + p * c));
            }
        }

        (a, b, c)
    }
}

impl<T, D> LineSearch<T, D> for GoldenSectionLineSearch
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        _f: T,
        _g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T {
        let resphi = T::from_f64((3.0_f64 - (5.0_f64).sqrt()) / 2.0).unwrap();
        let tol = T::from_f64(self.conf.tol).unwrap();

        let (mut a, b, mut c) = self.bracket_maximum(x, p, opt_prob);
        let mut x0 = b - resphi * (c - a);
        let mut x1 = a + resphi * (c - a);
        let mut f0 = opt_prob.evaluate(&(x + p * x0));
        let mut f1 = opt_prob.evaluate(&(x + p * x1));

        loop {
            if (c - a).abs() < tol {
                break (a + c) / T::from_f64(2.0).unwrap();
            }

            if f0 > f1 {
                c = x1;
                x1 = x0;
                f1 = f0;
                x0 = b - resphi * (c - a);
                f0 = opt_prob.evaluate(&(x + p * x0));
            } else {
                a = x0;
                x0 = x1;
                f0 = f1;
                x1 = a + resphi * (c - a);
                f1 = opt_prob.evaluate(&(x + p * x1));
            }
        }
    }
}

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

pub enum MoveType {
    RandomDrift,
    MALA,
}

pub struct MetropolisHastings<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub k: T,
    pub move_type: MoveType,
    pub prob: OptProb<T, D>,
    pub alpha: T,
    pub omega: T,
    pub mala_step_size: T,
}

impl<T, D> MetropolisHastings<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    pub fn new(
        prob: OptProb<T, D>,
        mala_step_size: T,
        alpha: T,
        omega: T,
        generic_x: OVector<T, D>,
    ) -> Self {
        let k = T::from_f64(1.38064852e-23).unwrap(); // Boltzmann constant

        let move_type = if prob.objective.gradient(&generic_x).is_some() {
            MoveType::MALA
        } else {
            MoveType::RandomDrift
        };

        MetropolisHastings {
            k,
            move_type,
            prob,
            mala_step_size,
            alpha,
            omega,
        }
    }

    pub fn local_move(
        &self,
        x_old: &OVector<T, D>,
        step_size: &OMatrix<T, D, D>,
        t: T,
    ) -> OVector<T, D> {
        match self.move_type {
            MoveType::MALA => {
                let grad = self
                    .prob
                    .objective
                    .gradient(x_old)
                    .expect("Gradient should be available for MALA");
                self.local_move_mala(x_old, &grad, t)
            }
            MoveType::RandomDrift => self.local_move_random_drift(x_old, step_size),
        }
    }

    fn local_move_random_drift(
        &self,
        x_old: &OVector<T, D>,
        step_size: &OMatrix<T, D, D>,
    ) -> OVector<T, D> {
        let mut rng = rand::rng();
        let mut x_new = x_old.clone();
        let random_vec =
            OVector::<T, D>::from_fn_generic(D::from_usize(x_old.len()), U1, |_, _| {
                T::from_f64(rng.random::<f64>()).unwrap()
            });
        x_new += random_vec.component_mul(&step_size.diagonal());
        x_new
    }

    fn local_move_mala(&self, x_old: &OVector<T, D>, grad: &OVector<T, D>, t: T) -> OVector<T, D> {
        let mut rng = rand::rng();
        let step = self.mala_step_size / t;
        let drift = grad * step;

        let noise = OVector::<T, D>::from_fn_generic(D::from_usize(x_old.len()), U1, |_, _| {
            T::from_f64(rng.sample::<f64, _>(StandardNormal)).unwrap()
        }) * (step * T::from_f64(2.0).unwrap()).sqrt();

        x_old + drift + noise
    }

    fn project(&self, x: &OVector<T, D>) -> OVector<T, D> {
        if let (Some(x_ub), Some(x_lb)) = (
            &self.prob.objective.x_upper_bound(x),
            &self.prob.objective.x_lower_bound(x),
        ) {
            x.component_mul(&(x_ub.clone() - x_lb.clone())) + x_lb.clone()
        } else {
            x.clone()
        }
    }

    pub fn accept_reject(
        &self,
        x_old: &OVector<T, D>,
        x_new: &OVector<T, D>,
        constraints_new: bool,
        t: T,
        t_swap: T,
    ) -> bool {
        // Reject if new solution violates constraints
        if !constraints_new {
            return false;
        }

        let diff = x_new - x_old;
        let delta_x = diff.dot(&diff).sqrt();

        let r: T;
        if t_swap > T::from_f64(0.0).unwrap() {
            // Pass in next temperature to signal global move
            let delta_t = (T::one() / t - T::one() / t_swap).powf(-T::one());
            let delta_f = self.prob.objective.f(&self.project(x_new))
                - self.prob.objective.f(&self.project(x_old));
            r = (delta_f / (self.k * delta_x * delta_t)).exp();
        } else {
            // Pass in negative anything to signal local move

            let delta_f = self.prob.objective.f(&self.project(x_new))
                - self.prob.objective.f(&self.project(x_old));

            // Correct asymmetry in proposal distribution if MALA
            let langevin_correction =
                if let Some(grad) = self.prob.objective.gradient(&self.project(x_old)) {
                    let proposal_grad = self.prob.objective.gradient(&self.project(x_new)).unwrap();
                    let grad_term = -((x_new - x_old - grad.clone() * self.mala_step_size / t)
                        .dot(&(x_new - x_old - grad.clone() * self.mala_step_size / t))
                        / (T::from_f64(4.0).unwrap() * self.mala_step_size / t))
                        + ((x_old - x_new - proposal_grad.clone() * self.mala_step_size / t).dot(
                            &(x_old - x_new - proposal_grad.clone() * self.mala_step_size / t),
                        ) / (T::from_f64(4.0).unwrap() * self.mala_step_size / t));
                    grad_term
                } else {
                    T::zero()
                };

            r = ((delta_f) / (self.k * delta_x * t) + langevin_correction).exp();
        }

        let mut rng = rand::rng();
        let u = T::from_f64(rng.random::<f64>()).unwrap();
        u < r
    }

    pub fn update_step_size(
        &self,
        step_size: &OMatrix<T, D, D>,
        x_old: &OVector<T, D>,
        x_new: &OVector<T, D>,
    ) -> OMatrix<T, D, D> {
        let r = OVector::<T, D>::from_fn_generic(D::from_usize(x_old.len()), U1, |i, _| {
            (x_new[i] - x_old[i]).abs()
        });
        let mut step_size_new = step_size.clone();
        for i in 0..x_old.len() {
            step_size_new[(i, i)] =
                (T::one() - self.alpha) * step_size[(i, i)] + self.alpha * self.omega * r[i];
        }
        step_size_new
    }
}

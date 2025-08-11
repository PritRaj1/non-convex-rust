use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::utils::alg_conf::pt_conf::UpdateConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

#[derive(Clone)]
pub enum MoveType {
    RandomDrift,
    MALA,
    PCN,
}

/// Poorly names struct for all update types
#[derive(Clone)]
pub struct MetropolisHastings<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    pub k: T, // Boltzmann constant
    pub move_type: MoveType,
    pub prob: OptProb<T, D>,
    pub mala_step_size: T,
    pub pcn_step_size: T,
    pub pcn_preconditioner: T,
}

impl<T, D> MetropolisHastings<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    pub fn new(prob: OptProb<T, D>, update_conf: &UpdateConf, generic_x: OVector<T, D>) -> Self {
        let k = T::from_f64(1.0).unwrap(); // Boltzmann constant

        // MALA needs gradients, PCN and Metropolis-Hastings don't
        let (move_type, mala_step_size, pcn_step_size, pcn_preconditioner) = match update_conf {
            UpdateConf::MetropolisHastings(conf) => {
                if prob.objective.gradient(&generic_x).is_some() {
                    let step_size = T::from_f64(0.01).unwrap();
                    (
                        MoveType::MALA,
                        step_size,
                        step_size,
                        T::from_f64(1.0).unwrap(),
                    )
                } else {
                    let step_size = T::from_f64(conf.random_walk_step_size).unwrap();
                    (
                        MoveType::RandomDrift,
                        step_size,
                        step_size,
                        T::from_f64(1.0).unwrap(),
                    )
                }
            }
            UpdateConf::MALA(conf) => {
                if prob.objective.gradient(&generic_x).is_some() {
                    let step_size = T::from_f64(conf.step_size).unwrap();
                    (
                        MoveType::MALA,
                        step_size,
                        step_size,
                        T::from_f64(1.0).unwrap(),
                    )
                } else {
                    // Fallback to Metropolis-Hastings if no gradient
                    let step_size = T::from_f64(0.1).unwrap();
                    (
                        MoveType::RandomDrift,
                        step_size,
                        step_size,
                        T::from_f64(1.0).unwrap(),
                    )
                }
            }
            UpdateConf::PCN(conf) => {
                if prob.objective.gradient(&generic_x).is_some() {
                    let step_size = T::from_f64(0.01).unwrap();
                    (
                        MoveType::MALA,
                        step_size,
                        step_size,
                        T::from_f64(1.0).unwrap(),
                    )
                } else {
                    let step_size = T::from_f64(conf.step_size).unwrap();
                    let preconditioner = T::from_f64(conf.preconditioner).unwrap();
                    (MoveType::PCN, step_size, step_size, preconditioner)
                }
            }
            UpdateConf::Auto(_) => {
                if prob.objective.gradient(&generic_x).is_some() {
                    let step_size = T::from_f64(0.01).unwrap();
                    (
                        MoveType::MALA,
                        step_size,
                        step_size,
                        T::from_f64(1.0).unwrap(),
                    )
                } else {
                    let step_size = T::from_f64(0.1).unwrap();
                    (
                        MoveType::RandomDrift,
                        step_size,
                        step_size,
                        T::from_f64(1.0).unwrap(),
                    )
                }
            }
        };

        MetropolisHastings {
            k,
            move_type,
            prob,
            mala_step_size,
            pcn_step_size,
            pcn_preconditioner,
        }
    }

    pub fn local_move(
        &mut self,
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
            MoveType::PCN => self.local_move_pcn(x_old, step_size),
            MoveType::RandomDrift => self.local_move_random_drift(x_old, step_size),
        }
    }

    fn local_move_random_drift(
        &mut self,
        x_old: &OVector<T, D>,
        step_size: &OMatrix<T, D, D>,
    ) -> OVector<T, D> {
        let mut x_new = x_old.clone();
        let random_vec =
            OVector::<T, D>::from_fn_generic(D::from_usize(x_old.len()), U1, |_, _| {
                T::from_f64(rand::rng().sample::<f64, _>(StandardNormal)).unwrap()
            });
        x_new += random_vec.component_mul(&step_size.diagonal());
        x_new
    }

    fn local_move_mala(
        &mut self,
        x_old: &OVector<T, D>,
        grad: &OVector<T, D>,
        t: T,
    ) -> OVector<T, D> {
        let t_for_step = T::from_f64(1.0).unwrap() - t;
        let step = self.mala_step_size * t_for_step.max(T::from_f64(0.01).unwrap());
        let drift = grad * step;

        let noise = OVector::<T, D>::from_fn_generic(D::from_usize(x_old.len()), U1, |_, _| {
            T::from_f64(rand::rng().sample::<f64, _>(StandardNormal)).unwrap()
        }) * (step * T::from_f64(2.0).unwrap()).sqrt();

        x_old + drift + noise
    }

    fn local_move_pcn(
        &mut self,
        x_old: &OVector<T, D>,
        step_size: &OMatrix<T, D, D>,
    ) -> OVector<T, D> {
        let beta = self.pcn_step_size;
        let beta = beta
            .max(T::from_f64(0.01).unwrap())
            .min(T::from_f64(0.99).unwrap());

        // First term: √(1-β²) X_n
        let sqrt_term = (T::from_f64(1.0).unwrap() - beta * beta).sqrt();
        let first_term = x_old * sqrt_term;

        // Second term: β ε_{n+1} where ε ~ N(0, C₀)
        let noise = OVector::<T, D>::from_fn_generic(D::from_usize(x_old.len()), U1, |_, _| {
            T::from_f64(rand::rng().sample::<f64, _>(StandardNormal)).unwrap()
        });

        let second_term = step_size * noise * beta;
        first_term + second_term
    }

    pub fn accept_reject(
        &mut self,
        x_old: &OVector<T, D>,
        x_new: &OVector<T, D>,
        constraints_new: bool,
        t: T,
    ) -> bool {
        if !constraints_new {
            return false;
        }

        let f_old = self.prob.evaluate(x_old);
        let f_new = self.prob.evaluate(x_new);
        let delta_e = f_new - f_old;

        if delta_e >= T::from_f64(0.0).unwrap() {
            return true; // Always accept uphill moves
        }

        // Boltzmann acceptance criterion: P = min(1, exp(ΔE / (k*(1-T))))
        let t_inverted = T::from_f64(1.0).unwrap() - t;
        let t_safe = t_inverted.max(T::from_f64(1e-10).unwrap());
        let acceptance_prob = (delta_e / (self.k * t_safe)).exp();
        rand::rng().random::<f64>() < acceptance_prob.to_f64().unwrap()
    }

    pub fn accept_replica_exchange<N>(
        &self,
        fitness_i: &OVector<T, N>,
        fitness_j: &OVector<T, N>,
        t_i: T,
        t_j: T,
    ) -> bool
    where
        N: Dim,
        DefaultAllocator: Allocator<N>,
    {
        // Calculate ensemble-level energy sums
        let mut total_energy_i = T::zero();
        let mut total_energy_j = T::zero();

        for &energy in fitness_i.iter() {
            total_energy_i += energy;
        }

        for &energy in fitness_j.iter() {
            total_energy_j += energy;
        }

        // Replica exchange acceptance criterion: P = min(1, exp((E_i - E_j) * (1/(k*(1-T_i)) - 1/(k*(1-T_j)))))
        // T=0 is hot, T=1 is cold, so invert temperatures
        let t_i_inverted = T::from_f64(1.0).unwrap() - t_i;
        let t_j_inverted = T::from_f64(1.0).unwrap() - t_j;
        let t_i_safe = t_i_inverted.max(T::from_f64(1e-10).unwrap());
        let t_j_safe = t_j_inverted.max(T::from_f64(1e-10).unwrap());

        let delta_beta = (T::from_f64(1.0).unwrap() / (self.k * t_i_safe))
            - (T::from_f64(1.0).unwrap() / (self.k * t_j_safe));
        let delta_e = total_energy_i - total_energy_j;

        let log_acceptance = delta_beta * delta_e;
        let acceptance_prob = if log_acceptance > T::from_f64(0.0).unwrap() {
            T::from_f64(1.0).unwrap()
        } else {
            log_acceptance.exp()
        };

        rand::rng().random::<f64>() < acceptance_prob.to_f64().unwrap()
    }

    pub fn update_step_size(
        &mut self,
        step_size: &OMatrix<T, D, D>,
        acceptance_rate: T,
        _t: T,
    ) -> OMatrix<T, D, D> {
        let target_rate = match self.move_type {
            MoveType::MALA => T::from_f64(0.574).unwrap(),
            MoveType::PCN => T::from_f64(0.574).unwrap(),
            MoveType::RandomDrift => T::from_f64(0.234).unwrap(),
        };

        let rate_diff = acceptance_rate - target_rate;
        let adjustment = if rate_diff > T::zero() {
            T::from_f64(1.1).unwrap()
        } else {
            T::from_f64(0.9).unwrap()
        };

        step_size * adjustment
    }
}

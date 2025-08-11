use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1, RealField, ComplexField, DimSub};
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
    T: FloatNum + RealField,
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
    T: FloatNum + RealField,
    D: Dim + DimSub<nalgebra::Const<1>>,
    DefaultAllocator: Allocator<D> + Allocator<D, D> + Allocator<<D as DimSub<nalgebra::Const<1>>>::Output>,
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
            MoveType::PCN => self.local_move_pcn(x_old, step_size),
            MoveType::RandomDrift => self.local_move_random_drift(x_old, step_size),
        }
    }

    fn local_move_random_drift(
        &self,
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
        &self,
        x_old: &OVector<T, D>,
        grad: &OVector<T, D>,
        t: T,
    ) -> OVector<T, D> {
        let t_for_step = T::from_f64(1.0).unwrap() - t;
        let step = self.mala_step_size * RealField::max(t_for_step, T::from_f64(0.01).unwrap());
        let drift = grad * step;

        let noise = OVector::<T, D>::from_fn_generic(D::from_usize(x_old.len()), U1, |_, _| {
            T::from_f64(rand::rng().sample::<f64, _>(StandardNormal)).unwrap()
        }) * ComplexField::sqrt(step * T::from_f64(2.0).unwrap());

        x_old + drift + noise
    }

    fn local_move_pcn(
        &self,
        x_old: &OVector<T, D>,
        covariance_matrix: &OMatrix<T, D, D>,
    ) -> OVector<T, D> {
        self.local_move_pcn_with_variance(x_old, covariance_matrix, T::from_f64(1.0).unwrap())
    }

    // Update for pCN: X'_{n+1} = √(1-β²) X_n + β ε_{n+1}
    // where ε_{n+1} ~ μ₀ (reference Gaussian measure)
    pub fn local_move_pcn_with_variance(
        &self,
        x_old: &OVector<T, D>,
        covariance_matrix: &OMatrix<T, D, D>,
        variance_param: T,
    ) -> OVector<T, D> {
        let beta = self.pcn_step_size;
        let beta = RealField::min(RealField::max(beta, T::from_f64(0.01).unwrap()), T::from_f64(0.99).unwrap());

        // First term: √(1-β²) X_n
        let sqrt_term = ComplexField::sqrt(T::from_f64(1.0).unwrap() - beta * beta);
        let first_term = x_old * sqrt_term;

        // Second term: β * ε_{n+1} where ε_{n+1} ~ N(0, C₀)
        let xi = OVector::<T, D>::from_fn_generic(D::from_usize(x_old.len()), U1, |_, _| {
            T::from_f64(rand::rng().sample::<f64, _>(StandardNormal)).unwrap()
        });

        let scaled_covariance = covariance_matrix * (variance_param * variance_param);
        
        // Use Cholesky: ε = L * ε where L is lower triangular, but fallback to eigendecomposition
        let xi_sample = if let Some(cholesky) = scaled_covariance.clone().cholesky() {
            
            cholesky.l() * xi
        } else {
            let eigen = scaled_covariance.symmetric_eigen();
            let eigenvalues = eigen.eigenvalues;
            let eigenvectors = eigen.eigenvectors;
            
            let sqrt_eigenvalues = OVector::<T, D>::from_fn_generic(
                D::from_usize(eigenvalues.len()),
                U1,
                |i, _| ComplexField::sqrt(RealField::max(eigenvalues[i], T::from_f64(1e-12).unwrap()))
            );
            
            let scaled_xi = xi.component_mul(&sqrt_eigenvalues);
            eigenvectors * scaled_xi
        };
        
        let second_term = xi_sample * beta;
        first_term + second_term
    }

    pub fn accept_reject(
        &self,
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
        let t_safe = RealField::max(t_inverted, T::from_f64(1e-10).unwrap());
        let acceptance_prob = ComplexField::exp(delta_e / (self.k * t_safe));
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
        let t_i_inverted = T::from_f64(1.0).unwrap() - t_i;
        let t_j_inverted = T::from_f64(1.0).unwrap() - t_j;
        let t_i_safe = RealField::max(t_i_inverted, T::from_f64(1e-10).unwrap());
        let t_j_safe = RealField::max(t_j_inverted, T::from_f64(1e-10).unwrap());

        let delta_beta = (T::from_f64(1.0).unwrap() / (self.k * t_i_safe))
            - (T::from_f64(1.0).unwrap() / (self.k * t_j_safe));
        let delta_e = total_energy_i - total_energy_j;

        let log_acceptance = delta_beta * delta_e;
        let acceptance_prob = if log_acceptance > T::from_f64(0.0).unwrap() {
            T::from_f64(1.0).unwrap()
        } else {
            ComplexField::exp(log_acceptance)
        };

        rand::rng().random::<f64>() < acceptance_prob.to_f64().unwrap()
    }

    /// Update step size based on Parks et al. (1990) approach
    pub fn update_step_size_parks(
        current_step_size: &OMatrix<T, D, D>,
        x_old: &OVector<T, D>,
        x_new: &OVector<T, D>,
        alpha: T, // Learning rate
        omega: T, // Scaling factor
    ) -> OMatrix<T, D, D> {
        // R = diag(|x_new - x_old|)
        let diff = x_new - x_old;
        let abs_diff = diff.map(|x| ComplexField::abs(x));
        let r = OMatrix::<T, D, D>::from_diagonal(&abs_diff);
        
        // Update: (1-alpha) * current + alpha * omega * r
        let one_minus_alpha = T::from_f64(1.0).unwrap() - alpha;
        current_step_size * one_minus_alpha + r * (alpha * omega)
    }
}

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::utils::config::SGAConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

pub struct SGAscent<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: SGAConf,
    pub opt_prob: OptProb<T, D>,
    pub x: OVector<T, D>,
    pub st: State<T, N, D>,
    velocity: OVector<T, D>,
    current_noise_std: f64,
    rng: StdRng,
}

impl<T, N, D> SGAscent<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<N>,
{
    pub fn new(
        conf: SGAConf,
        init_pop: OMatrix<T, U1, D>,
        opt_prob: OptProb<T, D>,
        seed: u64,
    ) -> Self {
        let init_x: OVector<T, D> = init_pop.row(0).transpose().into_owned();
        let best_f = opt_prob.evaluate(&init_x);
        let learning_rate = conf.learning_rate;
        let n = init_x.len();

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
            velocity: OVector::zeros_generic(D::from_usize(n), U1),
            current_noise_std: learning_rate,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for SGAscent<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        let mut gradient = self.opt_prob.objective.gradient(&self.x).unwrap();

        // Grad clip
        if self.conf.gradient_clip > 0.0 {
            let clip_norm = T::from_f64(self.conf.gradient_clip).unwrap();
            let grad_norm = gradient.dot(&gradient).sqrt();
            if grad_norm > clip_norm {
                gradient *= clip_norm / grad_norm;
            }
        }

        // Adaptive noise/decay
        self.current_noise_std *= self.conf.noise_decay;
        let noise_std = if self.conf.adaptive_noise {
            let grad_norm = gradient.dot(&gradient).sqrt().to_f64().unwrap();
            self.current_noise_std * (1.0 / (1.0 + grad_norm))
        } else {
            self.current_noise_std
        };

        let noise = OVector::<T, D>::from_iterator_generic(
            D::from_usize(self.x.len()),
            U1,
            (0..self.x.len()).map(|_| {
                T::from_f64(Normal::new(0.0, noise_std).unwrap().sample(&mut self.rng)).unwrap()
            }),
        );

        let noisy_gradient = gradient + noise;

        self.velocity = &self.velocity * T::from_f64(self.conf.momentum).unwrap()
            + &noisy_gradient * T::from_f64(self.conf.learning_rate).unwrap();

        self.x += &self.velocity;

        // Project back to bounds if infeasible
        if let (Some(lb), Some(ub)) = (
            self.opt_prob.objective.x_lower_bound(&self.x),
            self.opt_prob.objective.x_upper_bound(&self.x),
        ) {
            for i in 0..self.x.len() {
                self.x[i] = self.x[i].max(lb[i]).min(ub[i]);
            }
        }

        if self.opt_prob.is_feasible(&self.x) {
            let fitness = self.opt_prob.evaluate(&self.x);

            if fitness > self.st.best_f {
                self.st.best_f = fitness;
                self.st.best_x = self.x.clone();
            }
        }

        self.st.pop.row_mut(0).copy_from(&self.x.transpose());
        self.st.fitness[0] = self.opt_prob.evaluate(&self.x);
        self.st.constraints[0] = self.opt_prob.is_feasible(&self.x);
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}

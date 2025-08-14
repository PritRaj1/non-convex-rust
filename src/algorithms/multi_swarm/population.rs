use crate::algorithms::multi_swarm::swarm::Swarm;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, State};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rayon::prelude::*;

pub fn get_population<T, N, D>(swarms: &[Swarm<T, D>]) -> OMatrix<T, N, D>
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D> + Allocator<U1>,
{
    let total_particles: usize = swarms.iter().map(|s| s.particles.len()).sum();
    let dim = if let Some(first_swarm) = swarms.first() {
        if let Some(first_particle) = first_swarm.particles.first() {
            first_particle.position.len()
        } else {
            return OMatrix::<T, N, D>::zeros_generic(N::from_usize(0), D::from_usize(0));
        }
    } else {
        return OMatrix::<T, N, D>::zeros_generic(N::from_usize(0), D::from_usize(0));
    };

    let mut population =
        OMatrix::<T, N, D>::zeros_generic(N::from_usize(total_particles), D::from_usize(dim));

    let mut row_idx = 0;
    for swarm in swarms {
        for particle in &swarm.particles {
            for (col_idx, &value) in particle.position.iter().enumerate() {
                population[(row_idx, col_idx)] = value;
            }
            row_idx += 1;
        }
    }

    population
}

pub fn update_population_state<T, N, D>(
    state: &mut State<T, N, D>,
    swarms: &[Swarm<T, D>],
    opt_prob: &OptProb<T, D>,
) where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D> + Allocator<U1>,
{
    state.pop = get_population(swarms);

    let (fitness, constraints): (Vec<T>, Vec<bool>) = (0..state.pop.nrows())
        .into_par_iter()
        .map(|i| {
            let x = state.pop.row(i).transpose();
            let fit = opt_prob.evaluate(&x);
            let constr = opt_prob.is_feasible(&x);
            (fit, constr)
        })
        .unzip();

    state.fitness =
        OVector::<T, N>::from_vec_generic(N::from_usize(state.pop.nrows()), U1, fitness);
    state.constraints =
        OVector::<bool, N>::from_vec_generic(N::from_usize(state.pop.nrows()), U1, constraints);
}

pub fn find_best_solution<T, N, D>(
    population: &OMatrix<T, N, D>,
    opt_prob: &OptProb<T, D>,
) -> (OVector<T, D>, T)
where
    T: FloatNum + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator:
        Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D> + Allocator<U1>,
{
    let feasible_solutions: Vec<_> = (0..population.nrows())
        .into_par_iter()
        .filter_map(|i| {
            let x = population.row(i).transpose();
            if opt_prob.is_feasible(&x) {
                Some((x.clone(), opt_prob.evaluate(&x)))
            } else {
                None
            }
        })
        .collect();

    feasible_solutions
        .into_par_iter()
        .max_by(|(_, f1), (_, f2)| f1.partial_cmp(f2).unwrap())
        .unwrap_or_else(|| {
            let x = population.row(0).transpose();
            (x.clone(), opt_prob.evaluate(&x))
        })
}

use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rayon::prelude::*;

pub fn evaluate_samples<T, D>(
    samples: &[OVector<T, D>], // use slice
    mean: &OVector<T, D>,
    b_mat: &OMatrix<T, D, D>,
    d_vec: &OVector<T, D>,
    opt_prob: &OptProb<T, D>,
    sigma: T,
) -> Vec<(OVector<T, D>, T, bool)>
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    samples
        .par_iter()
        .map(|x| {
            let y = b_mat * &d_vec.component_mul(x);
            let mut sample = mean.clone();
            for i in 0..sample.len() {
                sample[i] += sigma * y[i];
            }

            if let (Some(lb), Some(ub)) = (
                opt_prob.objective.x_lower_bound(&sample),
                opt_prob.objective.x_upper_bound(&sample),
            ) {
                for i in 0..sample.len() {
                    sample[i] = sample[i].max(lb[i]).min(ub[i]);
                }
            }

            let fitness = opt_prob.evaluate(&sample);
            let constraint = opt_prob.is_feasible(&sample);
            (sample, fitness, constraint)
        })
        .collect()
}

pub fn update_arrays<T: FloatNum, N: Dim, D: Dim>(
    population: &mut OMatrix<T, N, D>,
    fitness: &mut OVector<T, N>,
    constraints: &mut OVector<bool, N>,
    results: &[(OVector<T, D>, T, bool)],
) where
    DefaultAllocator: Allocator<N, D> + Allocator<N> + Allocator<D> + Allocator<U1, D>,
{
    for (i, (x, f, c)) in results.iter().enumerate() {
        population.row_mut(i).copy_from(&x.transpose());
        fitness[i] = *f;
        constraints[i] = *c;
    }
}

pub fn sort<T, N>(
    fitness: &OVector<T, N>,
    constraints: &OVector<bool, N>,
    lambda: usize,
) -> Vec<usize>
where
    T: FloatNum,
    N: Dim,
    OVector<T, N>: Send + Sync,
    DefaultAllocator: Allocator<N>,
{
    let mut indices: Vec<usize> = (0..lambda).collect();
    indices.sort_by(|&i, &j| {
        let feasible_i = constraints[i];
        let feasible_j = constraints[j];
        match (feasible_i, feasible_j) {
            (true, true) => fitness[j].partial_cmp(&fitness[i]).unwrap(),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => fitness[j].partial_cmp(&fitness[i]).unwrap(),
        }
    });
    indices
}

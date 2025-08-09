use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector};
use rand::Rng;

use crate::utils::opt_prob::FloatNumber as FloatNum;

pub trait MutationStrategy<T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    fn generate_trial(
        &self,
        population: &OMatrix<T, N, D>,
        best_x: Option<&OVector<T, D>>,
        target_idx: usize,
        f: T,
        cr: T,
    ) -> OVector<T, D>;
}

pub struct Rand1Bin;
pub struct Best1Bin;
pub struct RandToBest1Bin;
pub struct Best2Bin;
pub struct Rand2Bin;

fn get_random_indices(count: usize, exclude: usize, pop_size: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    let mut indices = Vec::new();
    while indices.len() < count {
        let idx = rng.random_range(0..pop_size);
        if idx != exclude && !indices.contains(&idx) {
            indices.push(idx);
        }
    }
    indices
}

fn crossover<T, D>(donor: OVector<T, D>, target: OVector<T, D>, cr: T) -> OVector<T, D>
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    let mut rng = rand::rng();
    let dim = donor.len();
    let mut trial = target.clone();
    let j_rand = rng.random_range(0..dim);

    for j in 0..dim {
        if rng.random::<f64>() < cr.to_f64().unwrap() || j == j_rand {
            trial[j] = donor[j];
        }
    }
    trial
}

impl<T, N, D> MutationStrategy<T, N, D> for Rand1Bin
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    fn generate_trial(
        &self,
        population: &OMatrix<T, N, D>,
        _best_x: Option<&OVector<T, D>>,
        target_idx: usize,
        f: T,
        cr: T,
    ) -> OVector<T, D> {
        let indices = get_random_indices(3, target_idx, population.nrows());
        let x_r1 = population.row(indices[0]).transpose();
        let x_r2 = population.row(indices[1]).transpose();
        let x_r3 = population.row(indices[2]).transpose();

        let donor = x_r1 + (x_r2 - x_r3) * f;
        crossover(donor, population.row(target_idx).transpose(), cr)
    }
}

impl<T, N, D> MutationStrategy<T, N, D> for Best1Bin
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    fn generate_trial(
        &self,
        population: &OMatrix<T, N, D>,
        best_x: Option<&OVector<T, D>>,
        target_idx: usize,
        f: T,
        cr: T,
    ) -> OVector<T, D> {
        let best_x = best_x.expect("Best1Bin requires best_x");
        let indices = get_random_indices(2, target_idx, population.nrows());
        let x_r1 = population.row(indices[0]).transpose();
        let x_r2 = population.row(indices[1]).transpose();

        let donor = best_x + (x_r1 - x_r2) * f;
        crossover(donor, population.row(target_idx).transpose(), cr)
    }
}

impl<T, N, D> MutationStrategy<T, N, D> for RandToBest1Bin
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    fn generate_trial(
        &self,
        population: &OMatrix<T, N, D>,
        best_x: Option<&OVector<T, D>>,
        target_idx: usize,
        f: T,
        cr: T,
    ) -> OVector<T, D> {
        let best_x = best_x.expect("RandToBest1Bin requires best_x");
        let indices = get_random_indices(2, target_idx, population.nrows());
        let x_r1 = population.row(indices[0]).transpose();
        let x_r2 = population.row(indices[1]).transpose();
        let x_i = population.row(target_idx).transpose();

        let donor = x_i.clone() + (best_x - &x_i) * f + (x_r1 - x_r2) * f;
        crossover(donor, x_i, cr)
    }
}

impl<T, N, D> MutationStrategy<T, N, D> for Best2Bin
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    fn generate_trial(
        &self,
        population: &OMatrix<T, N, D>,
        best_x: Option<&OVector<T, D>>,
        target_idx: usize,
        f: T,
        cr: T,
    ) -> OVector<T, D> {
        let best_x = best_x.expect("Best2Bin requires best_x");
        let indices = get_random_indices(4, target_idx, population.nrows());
        let x_r1 = population.row(indices[0]).transpose();
        let x_r2 = population.row(indices[1]).transpose();
        let x_r3 = population.row(indices[2]).transpose();
        let x_r4 = population.row(indices[3]).transpose();

        let donor = best_x + (x_r1 + x_r2 - x_r3 - x_r4) * f;
        crossover(donor, population.row(target_idx).transpose(), cr)
    }
}

impl<T, N, D> MutationStrategy<T, N, D> for Rand2Bin
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    fn generate_trial(
        &self,
        population: &OMatrix<T, N, D>,
        _best_x: Option<&OVector<T, D>>,
        target_idx: usize,
        f: T,
        cr: T,
    ) -> OVector<T, D> {
        let indices = get_random_indices(5, target_idx, population.nrows());
        let x_r1 = population.row(indices[0]).transpose();
        let x_r2 = population.row(indices[1]).transpose();
        let x_r3 = population.row(indices[2]).transpose();
        let x_r4 = population.row(indices[3]).transpose();
        let x_r5 = population.row(indices[4]).transpose();

        let donor = x_r1 + (x_r2 + x_r3 - x_r4 - x_r5) * f;
        crossover(donor, population.row(target_idx).transpose(), cr)
    }
}

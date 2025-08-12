use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, DimSub, Dyn, OMatrix, OVector, RealField, U1,
};

use crate::utils::opt_prob::FloatNumber as FloatNum;

// Immutable parameters
#[derive(Debug)]
pub struct PathUpdateParams<'a, T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    pub ps: &'a mut OVector<T, D>,
    pub b_mat: &'a OMatrix<T, D, D>,
    pub d_vec: &'a OVector<T, D>,
    pub cs: T,
    pub mueff: T,
    pub generation: usize,
    pub chi_n: T,
    pub y: &'a OVector<T, D>,
    pub n: usize,
}

pub fn compute_y<T, D>(mean: &OVector<T, D>, old_mean: &OVector<T, D>, sigma: T) -> OVector<T, D>
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    let mut y = OVector::from_element_generic(D::from_usize(mean.len()), U1, T::zero());
    for i in 0..mean.len() {
        y[i] = (mean[i] - old_mean[i]) / sigma;
    }
    y
}

pub fn update_paths<T, D>(params: &mut PathUpdateParams<T, D>) -> bool
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    let d_inv = params.d_vec.map(|d| T::one() / d);
    let b_trans_y = params.b_mat.transpose() * params.y;
    let bdinvy = params.b_mat * &d_inv.component_mul(&b_trans_y);

    let cs_factor = T::sqrt(params.cs * (T::from_f64(2.0).unwrap() - params.cs) * params.mueff);

    // Update ps
    let mut ps_new = OVector::zeros_generic(D::from_usize(params.n), U1);
    for i in 0..params.n {
        ps_new[i] = (T::one() - params.cs) * params.ps[i] + cs_factor * bdinvy[i];
    }
    *params.ps = ps_new;

    // Update hsig
    let decay = T::one() - params.cs;
    let decay_pow = decay.powi(2 * params.generation as i32);
    let ps_norm = params.ps.dot(params.ps).sqrt();
    ps_norm / (T::sqrt(T::one() - decay_pow) * params.chi_n) < T::from_f64(1.4).unwrap()
}

#[derive(Debug)]
pub struct CovarianceUpdateParams<'a, T, N, D>
where
    T: FloatNum,
    N: Dim,
    D: Dim,
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<N, D> + Allocator<U1, D>,
{
    pub c_mat: &'a mut OMatrix<T, D, D>,
    pub b_mat: &'a mut OMatrix<T, D, D>,
    pub d_vec: &'a mut OVector<T, D>,
    pub pc: &'a mut OVector<T, D>,
    pub y: &'a OVector<T, D>,
    pub hsig: bool,
    pub indices: &'a [usize],
    pub old_mean: &'a OVector<T, D>,
    pub c1: T,
    pub cmu: T,
    pub cc: T,
    pub mueff: T,
    pub population: &'a OMatrix<T, N, D>,
    pub weights: &'a OVector<T, Dyn>,
    pub sigma: T,
    pub mu: usize,
    pub n: usize,
}

pub fn update_covariance<T: FloatNum + RealField, N: Dim, D>(
    params: &mut CovarianceUpdateParams<T, N, D>,
) where
    D: Dim + DimSub<nalgebra::Const<1>>,
    DefaultAllocator: Allocator<D, D>
        + Allocator<D>
        + Allocator<N, D>
        + Allocator<U1, D>
        + Allocator<<D as DimSub<nalgebra::Const<1>>>::Output>,
{
    let mut c_mat_new: OMatrix<T, D, D> =
        OMatrix::zeros_generic(D::from_usize(params.n), D::from_usize(params.n));
    let factor = T::one() - params.c1 - params.cmu;

    // Base update
    for i in 0..params.n {
        for j in 0..params.n {
            c_mat_new[(i, j)] = factor * params.c_mat[(i, j)];
        }
    }

    // Update pc
    let cc_factor =
        num_traits::Float::sqrt(params.cc * (T::from_f64(2.0).unwrap() - params.cc) * params.mueff);
    let hsig_t = if params.hsig { T::one() } else { T::zero() };

    // Update pc with single loop
    for i in 0..params.n {
        params.pc[i] = (T::one() - params.cc) * params.pc[i] + hsig_t * cc_factor * params.y[i];
    }

    // Rank-one update with single loop over upper triangle
    for i in 0..params.n {
        for j in i..params.n {
            let val = params.c1 * params.pc[i] * params.pc[j];
            c_mat_new[(i, j)] += val;
            if i != j {
                c_mat_new[(j, i)] += val;
            }
        }
    }

    // Rank-mu update
    for k in 0..params.mu {
        if k >= params.indices.len() || k >= params.weights.len() {
            continue;
        }
        let idx = params.indices[k];
        if idx >= params.population.nrows() {
            continue;
        }
        let w = params.weights[k];

        let mut y_k: OVector<T, D> = OVector::zeros_generic(D::from_usize(params.n), U1);
        for i in 0..params.n {
            if i < params.population.ncols() {
                y_k[i] = (params.population[(idx, i)] - params.old_mean[i]) / params.sigma;
            }
        }

        // Update c_mat_new with upper triangle only
        for i in 0..params.n {
            for j in i..params.n {
                let val = params.cmu * w * y_k[i] * y_k[j];
                c_mat_new[(i, j)] += val;
                if i != j {
                    c_mat_new[(j, i)] += val;
                }
            }
        }
    }

    *params.c_mat = c_mat_new;

    let eigen = params.c_mat.clone().symmetric_eigen();
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    for i in 0..params.n {
        params.d_vec[i] = nalgebra::ComplexField::sqrt(nalgebra::RealField::max(
            eigenvalues[i],
            T::from_f64(1e-20).unwrap(),
        ));
    }

    *params.b_mat = eigenvectors;

    // Restore C = BDB^T
    let d_mat: OMatrix<T, D, D> = OMatrix::from_diagonal(&params.d_vec.map(|x| x * x));
    let temp = &*params.b_mat * &d_mat;
    *params.c_mat = &temp * &params.b_mat.transpose();

    // Enforce symmetry (could be lost due to numerical errors)
    for i in 0..params.n {
        for j in i + 1..params.n {
            let avg = (params.c_mat[(i, j)] + params.c_mat[(j, i)]) * T::from_f64(0.5).unwrap();
            params.c_mat[(i, j)] = avg;
            params.c_mat[(j, i)] = avg;
        }
    }
}

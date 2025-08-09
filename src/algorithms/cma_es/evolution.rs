use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1};

use crate::utils::opt_prob::FloatNumber as FloatNum;

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

pub fn update_paths<T, D>(
    ps: &mut OVector<T, D>,
    b_mat: &OMatrix<T, D, D>,
    d_vec: &OVector<T, D>,
    cs: T,
    mueff: T,
    generation: usize,
    chi_n: T,
    y: &OVector<T, D>,
    n: usize,
) -> bool
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    let b_mat = &*b_mat;
    let d_inv = d_vec.map(|d| T::one() / d);
    let b_trans_y = b_mat.transpose() * y;
    let bdinvy = b_mat * &d_inv.component_mul(&b_trans_y);

    let cs_factor = T::sqrt(cs * (T::from_f64(2.0).unwrap() - cs) * mueff);

    // Update ps
    let mut ps_new = OVector::zeros_generic(D::from_usize(n), U1);
    for i in 0..n {
        ps_new[i] = (T::one() - cs) * ps[i] + cs_factor * bdinvy[i];
    }
    *ps = ps_new;

    // Update hsig
    let decay = T::one() - cs;
    let decay_pow = decay.powi(2 * generation as i32);
    let ps_norm = ps.dot(&ps).sqrt();
    ps_norm / (T::sqrt(T::one() - decay_pow) * chi_n) < T::from_f64(1.4).unwrap()
}

pub fn update_covariance<T: FloatNum, N: Dim, D: Dim>(
    c_mat: &mut OMatrix<T, D, D>,
    b_mat: &mut OMatrix<T, D, D>,
    d_vec: &mut OVector<T, D>,
    pc: &mut OVector<T, D>,
    y: &OVector<T, D>,
    hsig: bool,
    indices: &[usize],
    old_mean: &OVector<T, D>,
    c1: T,
    cmu: T,
    cc: T,
    mueff: T,
    population: &OMatrix<T, N, D>,
    weights: &OVector<T, Dyn>,
    sigma: T,
    mu: usize,
    n: usize,
) where
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<N, D> + Allocator<U1, D>,
{
    let mut c_mat_new: OMatrix<T, D, D> =
        OMatrix::zeros_generic(D::from_usize(n), D::from_usize(n));
    let factor = T::one() - c1 - cmu;

    // Base update
    for i in 0..n {
        for j in 0..n {
            c_mat_new[(i, j)] = factor * c_mat[(i, j)];
        }
    }

    // Update pc
    let cc_factor = T::sqrt(cc * (T::from_f64(2.0).unwrap() - cc) * mueff);
    let hsig_t = if hsig { T::one() } else { T::zero() };

    // Update pc with single loop
    for i in 0..n {
        pc[i] = (T::one() - cc) * pc[i] + hsig_t * cc_factor * y[i];
    }

    // Rank-one update with single loop over upper triangle
    for i in 0..n {
        for j in i..n {
            let val = c1 * pc[i] * pc[j];
            c_mat_new[(i, j)] += val;
            if i != j {
                c_mat_new[(j, i)] += val;
            }
        }
    }

    // Rank-mu update
    for k in 0..mu {
        if k >= indices.len() || k >= weights.len() {
            continue;
        }
        let idx = indices[k];
        if idx >= population.nrows() {
            continue;
        }
        let w = weights[k];

        let mut y_k: OVector<T, D> = OVector::zeros_generic(D::from_usize(n), U1);
        for i in 0..n {
            if i < population.ncols() {
                y_k[i] = (population[(idx, i)] - old_mean[i]) / sigma;
            }
        }

        // Update c_mat_new with upper triangle only
        for i in 0..n {
            for j in i..n {
                let val = cmu * w * y_k[i] * y_k[j];
                c_mat_new[(i, j)] += val;
                if i != j {
                    c_mat_new[(j, i)] += val;
                }
            }
        }
    }

    *c_mat = c_mat_new;

    // Symmetric power iteration with improvements
    let mut eigenvectors: Vec<OVector<T, D>> = Vec::with_capacity(n);
    let mut c_deflated = c_mat.clone();

    /* Could parallelize this with Rayon - power iteration is used because covariance is
    positive semi-definite and symmetric, and it's memory efficient + stable.
    */
    for i in 0..n {
        // Initialize random vector
        let mut v: OVector<T, D> = OVector::from_fn_generic(D::from_usize(n), U1, |_, _| {
            T::from_f64(rand::random::<f64>()).unwrap() * T::from_f64(2.0).unwrap() - T::one()
        });

        // Orthogonalize against previous eigenvectors - this prevents repetition of eigenvectors
        for prev_v in &eigenvectors {
            let proj = prev_v.dot(&v);
            for j in 0..n {
                v[j] = v[j] - proj * prev_v[j];
            }
        }

        // Normalize
        let v_norm = T::sqrt(v.dot(&v));
        for j in 0..n {
            v[j] = v[j] / v_norm;
        }

        // Power iteration with Rayleigh quotient to improve convergence speed
        let mut eigenvalue = T::zero();
        let mut prev_eigenvalue = T::neg_infinity();

        for _ in 0..20 {
            // Usually converges to machine precision in < 20 iterations - but might need to fiddle?
            let v_new = &c_deflated * &v;
            let norm = T::sqrt(v_new.dot(&v_new));

            if norm > T::from_f64(1e-10).unwrap() {
                // Normalize v_new into v
                for j in 0..n {
                    v[j] = v_new[j] / norm;
                }

                // Rayleigh quotient for faster convergence - break if eigenvalue is stable
                eigenvalue = v.dot(&(&c_deflated * &v));

                let diff = (eigenvalue - prev_eigenvalue).abs();
                if diff < T::from_f64(1e-12).unwrap() {
                    break;
                }
                prev_eigenvalue = eigenvalue;
            }
        }

        // Store eigenpair
        d_vec[i] = T::sqrt(T::max(eigenvalue.abs(), T::from_f64(1e-20).unwrap()));
        b_mat.set_column(i, &v);
        eigenvectors.push(v.clone());

        // Hotelling's deflation (more stable than simple deflation) - this may improve numerical stability
        let vv_t = &v * &v.transpose();
        c_deflated -= &vv_t * eigenvalue;
    }

    // Restore C = BDB^T
    let d_mat: OMatrix<T, D, D> = OMatrix::from_diagonal(&d_vec.map(|x| x * x));
    let temp = &*b_mat * &d_mat;
    *c_mat = &temp * &b_mat.transpose();

    // Enforce symmetry (could be lost due to numerical errors)
    for i in 0..n {
        for j in i + 1..n {
            let avg = (c_mat[(i, j)] + c_mat[(j, i)]) * T::from_f64(0.5).unwrap();
            c_mat[(i, j)] = avg;
            c_mat[(j, i)] = avg;
        }
    }
}

use crate::algorithms::tpe::kernels::KernelDensityEstimator;
use crate::utils::alg_conf::tpe_conf::AcquisitionType;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};

/// EI(x) = E[max(0, f(x) - f(x_best))], analytically simpler to use ratio
pub fn expected_improvement<T, D: Dim>(
    x: &OVector<T, D>,
    kde_l: &KernelDensityEstimator<T, D>,
    kde_g: &KernelDensityEstimator<T, D>,
    _kappa: T,
) -> T
where
    T: crate::utils::opt_prob::FloatNumber,
    DefaultAllocator: Allocator<D>,
{
    let p_l = kde_l.evaluate(x);
    let p_g = kde_g.evaluate(x);

    let epsilon = T::from_f64(1e-10).unwrap();
    if p_g <= epsilon {
        return T::zero();
    }

    let ratio = p_l / p_g;
    if ratio <= epsilon {
        return T::zero();
    }

    ratio
}

/// UCB(x) = f(x) + kappa * sqrt(2 * log(1 / delta) / p(x))
pub fn upper_confidence_bound<T, D: Dim>(
    x: &OVector<T, D>,
    kde_l: &KernelDensityEstimator<T, D>,
    kde_g: &KernelDensityEstimator<T, D>,
    kappa: T,
) -> T
where
    T: crate::utils::opt_prob::FloatNumber,
    DefaultAllocator: Allocator<D>,
{
    let p_l = kde_l.evaluate(x);
    let p_g = kde_g.evaluate(x);

    let epsilon = T::from_f64(1e-10).unwrap();
    if p_g <= epsilon {
        return T::zero();
    }

    let ratio = p_l / p_g;
    let exploration_term = kappa * (p_l + p_g).sqrt();

    ratio + exploration_term
}

/// PI(x) = P(f(x) > f(x_best))
pub fn probability_improvement<T, D: Dim>(
    x: &OVector<T, D>,
    kde_l: &KernelDensityEstimator<T, D>,
    kde_g: &KernelDensityEstimator<T, D>,
    _kappa: T,
) -> T
where
    T: crate::utils::opt_prob::FloatNumber,
    DefaultAllocator: Allocator<D>,
{
    let p_l = kde_l.evaluate(x);
    let p_g = kde_g.evaluate(x);

    let epsilon = T::from_f64(1e-10).unwrap();
    if p_g <= epsilon {
        return T::zero();
    }

    let ratio = p_l / p_g;
    if ratio <= epsilon {
        return T::zero();
    }

    ratio / (T::one() + ratio)
}

/// ES(x) = -p(x) * log(p(x)) - p(x) * log(p(x))
pub fn entropy_search<T, D: Dim>(
    x: &OVector<T, D>,
    kde_l: &KernelDensityEstimator<T, D>,
    kde_g: &KernelDensityEstimator<T, D>,
    _kappa: T,
) -> T
where
    T: crate::utils::opt_prob::FloatNumber,
    DefaultAllocator: Allocator<D>,
{
    let p_l = kde_l.evaluate(x);
    let p_g = kde_g.evaluate(x);

    let epsilon = T::from_f64(1e-10).unwrap();
    if p_g <= epsilon || p_l <= epsilon {
        return T::zero();
    }

    let total = p_l + p_g;
    let p_l_norm = p_l / total;
    let p_g_norm = p_g / total;

    // Entropy-based acquisition: prefer points that reduce uncertainty
    let entropy = -p_l_norm * p_l_norm.ln() - p_g_norm * p_g_norm.ln();
    let ratio = p_l / p_g;
    ratio * entropy
}

pub type AcquisitionFunctionPtr<T, D> =
    fn(&OVector<T, D>, &KernelDensityEstimator<T, D>, &KernelDensityEstimator<T, D>, T) -> T;

// Builder
pub fn get_acquisition_function<T, D: Dim>(
    acquisition_type: AcquisitionType,
) -> AcquisitionFunctionPtr<T, D>
where
    T: crate::utils::opt_prob::FloatNumber,
    DefaultAllocator: Allocator<D>,
{
    match acquisition_type {
        AcquisitionType::ExpectedImprovement => expected_improvement,
        AcquisitionType::UpperConfidenceBound => upper_confidence_bound,
        AcquisitionType::ProbabilityImprovement => probability_improvement,
        AcquisitionType::EntropySearch => entropy_search,
    }
}

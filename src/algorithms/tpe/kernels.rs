use crate::utils::alg_conf::tpe_conf::{BandwidthConf, BandwidthMethod};
use crate::utils::opt_prob::FloatNumber as FloatNum;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum KernelType {
    Gaussian,
    Epanechnikov,
    Uniform,
}

impl KernelType {
    pub fn evaluate<T: FloatNum>(&self, x: T, bandwidth: T) -> T {
        match self {
            KernelType::Gaussian => {
                let z = x / bandwidth;
                let factor = T::from_f64(1.0 / (2.0 * std::f64::consts::PI).sqrt()).unwrap();
                factor * (-T::from_f64(0.5).unwrap() * z * z).exp() / bandwidth
            }
            KernelType::Epanechnikov => {
                let z = x / bandwidth;
                let abs_z = z.abs();
                if abs_z <= T::one() {
                    let factor = T::from_f64(0.75).unwrap();
                    factor * (T::one() - z * z) / bandwidth
                } else {
                    T::zero()
                }
            }
            KernelType::Uniform => {
                let z = x / bandwidth;
                if z.abs() <= T::from_f64(0.5).unwrap() {
                    T::one() / bandwidth
                } else {
                    T::zero()
                }
            }
        }
    }
}

pub fn create_kernel(kernel_type: &str) -> KernelType {
    match kernel_type.to_lowercase().as_str() {
        "gaussian" => KernelType::Gaussian,
        "epanechnikov" => KernelType::Epanechnikov,
        "uniform" => KernelType::Uniform,
        _ => KernelType::Gaussian, // Default to Gaussian
    }
}

pub struct KernelDensityEstimator<T: FloatNum, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    observations: Vec<OVector<T, D>>,
    bandwidths: Vec<T>,
    kernel: KernelType,
    bandwidth_conf: BandwidthConf,
}

impl<T: FloatNum, D: Dim> KernelDensityEstimator<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(observations: Vec<OVector<T, D>>, kernel: KernelType) -> Self {
        let _dim = observations.first().map(|x| x.len()).unwrap_or(0);
        let bandwidths = Self::compute_bandwidths(&observations, &BandwidthConf::default());

        Self {
            observations,
            bandwidths,
            kernel,
            bandwidth_conf: BandwidthConf::default(),
        }
    }

    pub fn new_with_config(
        observations: Vec<OVector<T, D>>,
        kernel: KernelType,
        bandwidth_conf: BandwidthConf,
    ) -> Self {
        let bandwidths = Self::compute_bandwidths(&observations, &bandwidth_conf);

        Self {
            observations,
            bandwidths,
            kernel,
            bandwidth_conf,
        }
    }

    pub fn evaluate(&self, x: &OVector<T, D>) -> T {
        if self.observations.is_empty() {
            return T::zero();
        }

        let mut density = T::zero();
        let n = self.observations.len();

        for (i, obs) in self.observations.iter().enumerate() {
            let bandwidth = self.bandwidths[i];
            let mut distance = T::zero();

            for j in 0..obs.len() {
                let diff = (obs[j] - x[j]) / bandwidth;
                distance += diff * diff;
            }
            distance = distance.sqrt();

            density += self.kernel.evaluate(distance, bandwidth);
        }

        density / T::from_usize(n).unwrap()
    }

    pub fn fit(&mut self, new_observations: &[OVector<T, D>]) {
        if self.should_recompute_bandwidths(new_observations) {
            self.observations.extend_from_slice(new_observations);
            self.bandwidths = Self::compute_bandwidths(&self.observations, &self.bandwidth_conf);
        } else {
            self.observations.extend_from_slice(new_observations);

            let new_count = new_observations.len();
            if new_count > 0 {
                let default_bandwidth = if !self.bandwidths.is_empty() {
                    self.bandwidths[0]
                } else {
                    T::from_f64(1.0).unwrap()
                };
                self.bandwidths.extend(vec![default_bandwidth; new_count]);
            }
        }
    }

    // Recompute when enough new observations
    fn should_recompute_bandwidths(&self, new_observations: &[OVector<T, D>]) -> bool {
        if self.observations.is_empty() {
            return true;
        }

        let new_count = new_observations.len();
        let existing_count = self.observations.len();

        if new_count > 0
            && (new_count as f64 / existing_count as f64) > self.bandwidth_conf.cache_threshold
        {
            return true;
        }

        if existing_count < self.bandwidth_conf.min_observations {
            return true;
        }

        false
    }

    pub fn add_observation(&mut self, observation: OVector<T, D>) {
        self.observations.push(observation);
        self.bandwidths = Self::compute_bandwidths(&self.observations, &self.bandwidth_conf);
    }

    pub fn update_observations(&mut self, new_observations: Vec<OVector<T, D>>) {
        self.observations = new_observations;
        self.bandwidths = Self::compute_bandwidths(&self.observations, &self.bandwidth_conf);
    }

    pub fn get_bandwidths(&self) -> &[T] {
        &self.bandwidths
    }

    pub fn set_bandwidth_conf(&mut self, bandwidth_conf: BandwidthConf) {
        self.bandwidth_conf = bandwidth_conf;
        self.bandwidths = Self::compute_bandwidths(&self.observations, &self.bandwidth_conf);
    }

    pub fn update_bandwidths(&mut self) {
        self.bandwidths = Self::compute_bandwidths(&self.observations, &self.bandwidth_conf);
    }

    fn compute_bandwidths(
        observations: &[OVector<T, D>],
        bandwidth_conf: &BandwidthConf,
    ) -> Vec<T> {
        match bandwidth_conf.method {
            BandwidthMethod::Silverman => Self::compute_silverman_bandwidths(observations),
            BandwidthMethod::CrossValidation => {
                Self::compute_cv_bandwidths(observations, bandwidth_conf)
            }
            BandwidthMethod::Adaptive => Self::compute_adaptive_bandwidths(observations),
            BandwidthMethod::LikelihoodBased => Self::compute_likelihood_bandwidths(observations),
        }
    }

    fn compute_silverman_bandwidths(observations: &[OVector<T, D>]) -> Vec<T> {
        if observations.is_empty() {
            return vec![];
        }

        let n = observations.len();
        let dim = observations[0].len();

        // Silverman's rule of thumb: h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * σ
        let factor =
            T::from_f64((4.0 / (dim as f64 + 2.0)).powf(1.0 / (dim as f64 + 4.0))).unwrap();
        let n_factor = T::from_f64((n as f64).powf(-1.0 / (dim as f64 + 4.0))).unwrap();

        let mut bandwidths = Vec::with_capacity(n);

        for obs in observations {
            let mut variance = T::zero();
            for &val in obs.iter() {
                variance += val * val;
            }
            variance /= T::from_usize(dim).unwrap();
            let std_dev = variance.sqrt();

            let bandwidth = factor * n_factor * std_dev;
            bandwidths.push(bandwidth.max(T::from_f64(1e-6).unwrap()));
        }

        bandwidths
    }

    // Cross-validation to find best bandwidth
    fn compute_cv_bandwidths(
        observations: &[OVector<T, D>],
        _bandwidth_conf: &BandwidthConf,
    ) -> Vec<T> {
        if observations.len() < 2 {
            return Self::compute_silverman_bandwidths(observations);
        }

        let mut bandwidths = Vec::with_capacity(observations.len());

        // Golden-section search, (similar to L-BFGS)
        for i in 0..observations.len() {
            let mut a = T::from_f64(0.01).unwrap();
            let mut b = T::from_f64(5.0).unwrap();
            let inv_phi = T::from_f64((3.0_f64 - (5.0_f64).sqrt()) / 2.0).unwrap();
            let tolerance = T::from_f64(1e-4).unwrap();

            let mut x1 = b - inv_phi * (b - a);
            let mut x2 = a + inv_phi * (b - a);

            let mut f1 = Self::compute_cv_score(observations, i, x1);
            let mut f2 = Self::compute_cv_score(observations, i, x2);

            while (b - a).abs() > tolerance {
                if f1 > f2 {
                    b = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = b - inv_phi * (b - a);
                    f1 = Self::compute_cv_score(observations, i, x1);
                } else {
                    a = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = a + inv_phi * (b - a);
                    f2 = Self::compute_cv_score(observations, i, x2);
                }
            }

            let best_bandwidth = if f1 > f2 { x1 } else { x2 };

            bandwidths.push(best_bandwidth);
        }

        bandwidths
    }

    fn find_best_bandwidth_cv(
        observations: &[OVector<T, D>],
        target_idx: usize,
        test_bandwidth: T,
    ) -> T {
        let mut score = T::zero();
        let n = observations.len();

        for i in 0..n {
            if i == target_idx {
                continue;
            }

            let obs = &observations[i];
            let target = &observations[target_idx];

            let mut distance = T::zero();
            for j in 0..obs.len() {
                let diff = (obs[j] - target[j]) / test_bandwidth;
                distance += diff * diff;
            }
            distance = distance.sqrt();

            let kernel_val = (-T::from_f64(0.5).unwrap() * distance * distance).exp();
            score += kernel_val;
        }

        score
    }

    fn compute_cv_score(observations: &[OVector<T, D>], target_idx: usize, bandwidth: T) -> T {
        Self::find_best_bandwidth_cv(observations, target_idx, bandwidth)
    }

    fn compute_adaptive_bandwidths(observations: &[OVector<T, D>]) -> Vec<T> {
        let mut bandwidths = Self::compute_silverman_bandwidths(observations); // Start with Silverman bandwidths

        if observations.len() < 2 {
            return bandwidths;
        }

        // Compute all local densities first to avoid borrowing conflicts
        let local_densities: Vec<T> = (0..observations.len())
            .map(|i| Self::compute_local_density(observations, i, &bandwidths))
            .collect();

        // Now update bandwidths using the pre-computed densities
        for (i, bandwidth) in bandwidths.iter_mut().enumerate() {
            let adaptation_factor = T::one() / (local_densities[i] + T::from_f64(1e-6).unwrap());
            *bandwidth *= adaptation_factor.sqrt();
        }

        bandwidths
    }

    fn compute_likelihood_bandwidths(observations: &[OVector<T, D>]) -> Vec<T> {
        let mut bandwidths = Self::compute_silverman_bandwidths(observations); // Start with Silverman bandwidths

        if observations.len() < 2 {
            return bandwidths;
        }

        for (i, bandwidth) in bandwidths.iter_mut().enumerate() {
            *bandwidth = Self::optimize_bandwidth_likelihood(observations, i, *bandwidth);
        }

        bandwidths
    }

    fn optimize_bandwidth_likelihood(
        observations: &[OVector<T, D>],
        target_idx: usize,
        initial_h: T,
    ) -> T {
        if observations.len() < 2 {
            return initial_h;
        }

        // Golden section search similar to L-BFGS
        let inv_phi = T::from_f64((3.0_f64 - (5.0_f64).sqrt()) / 2.0).unwrap();
        let tolerance = T::from_f64(1e-6).unwrap();

        let mut a = initial_h * T::from_f64(0.1).unwrap();
        let mut b = initial_h * T::from_f64(10.0).unwrap();

        let mut x1 = b - inv_phi * (b - a);
        let mut x2 = a + inv_phi * (b - a);

        let mut f1 = Self::compute_likelihood(observations, target_idx, x1);
        let mut f2 = Self::compute_likelihood(observations, target_idx, x2);

        while (b - a).abs() > tolerance {
            if f1 > f2 {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = b - inv_phi * (b - a);
                f1 = Self::compute_likelihood(observations, target_idx, x1);
            } else {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + inv_phi * (b - a);
                f2 = Self::compute_likelihood(observations, target_idx, x2);
            }
        }

        // Midpoint of the final interval
        (a + b) / T::from_f64(2.0).unwrap()
    }

    fn compute_likelihood(observations: &[OVector<T, D>], target_idx: usize, bandwidth: T) -> T {
        let mut likelihood = T::zero();
        let n = observations.len();

        for i in 0..n {
            if i == target_idx {
                continue;
            }

            let obs = &observations[i];
            let target = &observations[target_idx];

            let mut distance = T::zero();
            for j in 0..obs.len() {
                let diff = (obs[j] - target[j]) / bandwidth;
                distance += diff * diff;
            }
            distance = distance.sqrt();

            let kernel_val = (-T::from_f64(0.5).unwrap() * distance * distance).exp();
            likelihood += kernel_val.ln();
        }

        likelihood
    }

    fn compute_local_density(
        observations: &[OVector<T, D>],
        target_idx: usize,
        bandwidths: &[T],
    ) -> T {
        let mut density = T::zero();
        let n = observations.len();

        for i in 0..n {
            if i == target_idx {
                continue;
            }

            let obs = &observations[i];
            let target = &observations[target_idx];
            let bandwidth = bandwidths[i];

            // Simple distance-based density estimate
            let mut distance = T::zero();
            for j in 0..obs.len() {
                let diff = (obs[j] - target[j]) / bandwidth;
                distance += diff * diff;
            }
            distance = distance.sqrt();
            let kernel_val = (-T::from_f64(0.5).unwrap() * distance * distance).exp();
            density += kernel_val;
        }

        density / T::from_usize(n - 1).unwrap()
    }
}

use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};

pub struct BoundsCache<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub lower_bounds: OVector<T, D>,
    pub upper_bounds: OVector<T, D>,
    pub cached: bool,
}

impl<T, D> BoundsCache<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub fn new(dimension: usize) -> Self {
        Self {
            lower_bounds: OVector::<T, D>::from_element_generic(
                D::from_usize(dimension),
                U1,
                T::from_f64(-10.0).unwrap(),
            ),
            upper_bounds: OVector::<T, D>::from_element_generic(
                D::from_usize(dimension),
                U1,
                T::from_f64(10.0).unwrap(),
            ),
            cached: false,
        }
    }

    pub fn get_bounds(
        &mut self,
        opt_prob: &OptProb<T, D>,
        sample_individual: &OVector<T, D>,
    ) -> (T, T) {
        if !self.cached {
            let lower_bounds = opt_prob
                .objective
                .x_lower_bound(sample_individual)
                .unwrap_or_else(|| self.lower_bounds.clone());
            let upper_bounds = opt_prob
                .objective
                .x_upper_bound(sample_individual)
                .unwrap_or_else(|| self.upper_bounds.clone());

            self.lower_bounds = lower_bounds.clone();
            self.upper_bounds = upper_bounds.clone();
            self.cached = true;
        }

        (self.lower_bounds[0], self.upper_bounds[0])
    }

    pub fn reset(&mut self) {
        self.cached = false;
    }
}

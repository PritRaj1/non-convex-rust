use crate::utils::opt_prob::FloatNumber as FloatNum;

pub trait CoolingSchedule<T: FloatNum> {
    fn temperature(&self, initial_temp: T, iteration: usize, cooling_rate: T) -> T;
    fn reheat(&self, initial_temp: T) -> T;
}

pub struct ExponentialCooling;

impl<T: FloatNum> CoolingSchedule<T> for ExponentialCooling {
    fn temperature(&self, initial_temp: T, iteration: usize, cooling_rate: T) -> T {
        initial_temp * cooling_rate.powi(iteration as i32)
    }

    fn reheat(&self, initial_temp: T) -> T {
        initial_temp * T::from_f64(0.8).unwrap()
    }
}

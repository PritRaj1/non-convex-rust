use crate::utils::opt_prob::FloatNumber as FloatNum;

pub trait CoolingSchedule<T: FloatNum> {
    fn temperature(&self, initial_temp: T, iteration: usize, cooling_rate: T) -> T;
    fn reheat(&self, initial_temp: T) -> T;
    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        cooling_rate: T,
        success_rate: f64,
    ) -> T;
}

pub struct ExponentialCooling;

impl<T: FloatNum> CoolingSchedule<T> for ExponentialCooling {
    fn temperature(&self, initial_temp: T, iteration: usize, cooling_rate: T) -> T {
        initial_temp * cooling_rate.powi(iteration as i32)
    }

    fn reheat(&self, initial_temp: T) -> T {
        initial_temp * T::from_f64(0.8).unwrap()
    }

    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        cooling_rate: T,
        success_rate: f64,
    ) -> T {
        let base_temp = self.temperature(initial_temp, iteration, cooling_rate);
        if success_rate < 0.2 {
            base_temp * T::from_f64(1.2).unwrap() // Increase when low success, explore more
        } else if success_rate > 0.6 {
            base_temp * T::from_f64(0.9).unwrap() // Decrease when high success, exploit more
        } else {
            base_temp
        }
    }
}

pub struct LogarithmicCooling;

impl<T: FloatNum> CoolingSchedule<T> for LogarithmicCooling {
    fn temperature(&self, initial_temp: T, iteration: usize, _cooling_rate: T) -> T {
        initial_temp / T::from_f64(1.0 + (iteration as f64).ln()).unwrap()
    }

    fn reheat(&self, initial_temp: T) -> T {
        initial_temp * T::from_f64(0.7).unwrap()
    }

    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        _cooling_rate: T,
        success_rate: f64,
    ) -> T {
        let base_temp = self.temperature(initial_temp, iteration, T::one());
        if success_rate < 0.2 {
            base_temp * T::from_f64(1.3).unwrap()
        } else if success_rate > 0.6 {
            base_temp * T::from_f64(0.85).unwrap()
        } else {
            base_temp
        }
    }
}

pub struct CauchyCooling;

impl<T: FloatNum> CoolingSchedule<T> for CauchyCooling {
    fn temperature(&self, initial_temp: T, iteration: usize, _cooling_rate: T) -> T {
        initial_temp / T::from_f64(1.0 + iteration as f64).unwrap()
    }

    fn reheat(&self, initial_temp: T) -> T {
        initial_temp * T::from_f64(0.75).unwrap()
    }

    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        _cooling_rate: T,
        success_rate: f64,
    ) -> T {
        let base_temp = self.temperature(initial_temp, iteration, T::one());
        if success_rate < 0.2 {
            base_temp * T::from_f64(1.4).unwrap()
        } else if success_rate > 0.6 {
            base_temp * T::from_f64(0.8).unwrap()
        } else {
            base_temp
        }
    }
}

pub struct AdaptiveCooling;

// Start with exponential, then adapt based on iter
impl<T: FloatNum> CoolingSchedule<T> for AdaptiveCooling {
    fn temperature(&self, initial_temp: T, iteration: usize, cooling_rate: T) -> T {
        if iteration < 100 {
            initial_temp * cooling_rate.powi(iteration as i32)
        } else {
            initial_temp * cooling_rate.powi((iteration / 2) as i32)
        }
    }

    fn reheat(&self, initial_temp: T) -> T {
        initial_temp * T::from_f64(0.6).unwrap()
    }

    fn adaptive_temperature(
        &self,
        initial_temp: T,
        iteration: usize,
        cooling_rate: T,
        success_rate: f64,
    ) -> T {
        let base_temp = self.temperature(initial_temp, iteration, cooling_rate);

        let adaptation_factor = if success_rate < 0.1 {
            T::from_f64(2.0).unwrap() // Significant increase for very low success, explore more
        } else if success_rate < 0.3 {
            T::from_f64(1.5).unwrap() // Moderate increase for low success, explore more
        } else if success_rate > 0.7 {
            T::from_f64(0.7).unwrap() // Decrease for high success, exploit more
        } else {
            T::one()
        };

        base_temp * adaptation_factor
    }
}

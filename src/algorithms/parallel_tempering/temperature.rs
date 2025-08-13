use crate::utils::opt_prob::FloatNumber as FloatNum;
use nalgebra::RealField;

pub struct PowerLawScheduler<T>
where
    T: FloatNum + RealField,
{
    p_schedule: Vec<T>,
    num_replicas: usize,
    current_iter: usize,
}

impl<T> PowerLawScheduler<T>
where
    T: FloatNum + RealField,
{
    pub fn new(
        power_law_init: f64,
        power_law_final: f64,
        power_law_cycles: f64,
        num_replicas: usize,
        max_iter: usize,
    ) -> Self {
        // Power law schedule for cyclic annealing
        let p_schedule: Vec<T> = (0..=max_iter)
            .map(|i| {
                let x = 2.0
                    * std::f64::consts::PI
                    * (power_law_cycles + 0.5)
                    * (i as f64 / max_iter as f64);
                let p_current =
                    power_law_init + (power_law_final - power_law_init) * 0.5 * (1.0 - x.cos());
                T::from_f64(p_current).unwrap()
            })
            .collect();

        Self {
            p_schedule,
            num_replicas,
            current_iter: 1,
        }
    }

    /// Temperature should be in [0, 1] range, with replica 0 being hottest (0) and highest being coldest (1)
    pub fn get_temperature(&self, replica_idx: usize) -> T {
        let schedule_idx = (self.current_iter - 1).min(self.p_schedule.len() - 1);
        let power = self.p_schedule[schedule_idx].to_f64().unwrap();
        let temp = (replica_idx as f64 / (self.num_replicas - 1) as f64).powf(power);

        let temp_clamped = temp.clamp(1e-10, 1.0 - 1e-10);
        T::from_f64(temp_clamped).unwrap()
    }

    pub fn get_all_temperatures(&self) -> Vec<T> {
        (0..self.num_replicas)
            .map(|i| self.get_temperature(i))
            .collect()
    }

    pub fn update_iteration(&mut self, iter: usize) {
        self.current_iter = iter;
    }

    pub fn num_replicas(&self) -> usize {
        self.num_replicas
    }
}

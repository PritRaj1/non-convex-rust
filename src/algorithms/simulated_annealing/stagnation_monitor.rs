use crate::utils::opt_prob::FloatNumber as FloatNum;
use std::collections::VecDeque;

pub struct SAStagnationMonitor<T>
where
    T: FloatNum + Send + Sync,
{
    stagnation_counter: usize,
    last_best_fitness: T,
    improvement_threshold: T,
    improvement_history: VecDeque<f64>,
    success_history: VecDeque<bool>,
    temperature_history: VecDeque<f64>,
    step_size_history: VecDeque<f64>,
}

impl<T> SAStagnationMonitor<T>
where
    T: FloatNum + Send + Sync,
{
    pub fn new(improvement_threshold: T, initial_fitness: T) -> Self {
        Self {
            stagnation_counter: 0,
            last_best_fitness: initial_fitness,
            improvement_threshold,
            improvement_history: VecDeque::with_capacity(20),
            success_history: VecDeque::with_capacity(20),
            temperature_history: VecDeque::with_capacity(20),
            step_size_history: VecDeque::with_capacity(20),
        }
    }

    pub fn check_stagnation(&mut self, current_fitness: T, temperature: T, step_size: T) {
        let improvement = current_fitness - self.last_best_fitness;

        let improvement_f64 = improvement.to_f64().unwrap_or(0.0);
        self.improvement_history.push_back(improvement_f64);
        if self.improvement_history.len() > 20 {
            self.improvement_history.pop_front();
        }

        let success = improvement > self.improvement_threshold;
        self.success_history.push_back(success);
        if self.success_history.len() > 20 {
            self.success_history.pop_front();
        }

        let temp_f64 = temperature.to_f64().unwrap_or(0.0);
        self.temperature_history.push_back(temp_f64);
        if self.temperature_history.len() > 20 {
            self.temperature_history.pop_front();
        }

        let step_f64 = step_size.to_f64().unwrap_or(0.0);
        self.step_size_history.push_back(step_f64);
        if self.step_size_history.len() > 20 {
            self.step_size_history.pop_front();
        }

        if improvement < self.improvement_threshold {
            self.stagnation_counter += 1;
        } else {
            self.stagnation_counter = 0;
        }

        self.last_best_fitness = current_fitness;
    }

    pub fn stagnation_counter(&self) -> usize {
        self.stagnation_counter
    }

    pub fn is_stagnated(&self) -> bool {
        self.stagnation_counter > 20
    }

    pub fn get_performance_stats(&self) -> (f64, f64, f64, f64) {
        let avg_improvement = if self.improvement_history.len() > 5 {
            self.improvement_history.iter().sum::<f64>() / self.improvement_history.len() as f64
        } else {
            0.0
        };

        let success_rate = if !self.success_history.is_empty() {
            self.success_history.iter().filter(|&&x| x).count() as f64
                / self.success_history.len() as f64
        } else {
            0.0
        };

        let stagnation_level = self.stagnation_counter as f64;

        let avg_temperature = if !self.temperature_history.is_empty() {
            self.temperature_history.iter().sum::<f64>() / self.temperature_history.len() as f64
        } else {
            0.0
        };

        (
            avg_improvement,
            success_rate,
            stagnation_level,
            avg_temperature,
        )
    }

    pub fn should_restart(&self, restart_threshold: usize) -> bool {
        self.stagnation_counter > restart_threshold
    }

    pub fn get_adaptation_suggestions(&self) -> (f64, f64) {
        let success_rate = if !self.success_history.is_empty() {
            self.success_history.iter().filter(|&&x| x).count() as f64
                / self.success_history.len() as f64
        } else {
            0.0
        };

        let avg_improvement = if self.improvement_history.len() > 5 {
            self.improvement_history.iter().sum::<f64>() / self.improvement_history.len() as f64
        } else {
            0.0
        };

        let temp_factor = if success_rate < 0.2 {
            1.5 // Increase temperature for more exploration
        } else if success_rate > 0.6 && avg_improvement > 1e-4 {
            0.8 // Decrease temperature for more exploitation
        } else {
            1.0
        };

        let step_factor = if success_rate < 0.2 {
            1.3 // Increase step size for more exploration
        } else if success_rate > 0.6 {
            0.9 // Decrease step size for more exploitation
        } else {
            1.0
        };

        (temp_factor, step_factor)
    }
}

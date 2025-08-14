use crate::utils::opt_prob::FloatNumber as FloatNum;
use std::collections::VecDeque;

pub struct StagnationMonitor<T>
where
    T: FloatNum + Send + Sync,
{
    stagnation_counter: usize,
    last_best_fitness: T,
    improvement_threshold: T,
    generation_improvements: VecDeque<f64>,
    success_history: VecDeque<bool>,
}

impl<T> StagnationMonitor<T>
where
    T: FloatNum + Send + Sync,
{
    pub fn new(improvement_threshold: T, initial_fitness: T) -> Self {
        Self {
            stagnation_counter: 0,
            last_best_fitness: initial_fitness,
            improvement_threshold,
            generation_improvements: VecDeque::with_capacity(20),
            success_history: VecDeque::with_capacity(20),
        }
    }

    pub fn check_stagnation(&mut self, current_fitness: T) {
        let improvement = current_fitness - self.last_best_fitness;

        let improvement_f64 = improvement.to_f64().unwrap_or(0.0);
        self.generation_improvements.push_back(improvement_f64);
        if self.generation_improvements.len() > 20 {
            self.generation_improvements.pop_front();
        }

        let success = improvement > self.improvement_threshold;
        self.success_history.push_back(success);
        if self.success_history.len() > 20 {
            self.success_history.pop_front();
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

    pub fn get_performance_stats(&self) -> (f64, f64, f64) {
        let avg_improvement = if self.generation_improvements.len() > 5 {
            self.generation_improvements.iter().sum::<f64>()
                / self.generation_improvements.len() as f64
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

        (avg_improvement, success_rate, stagnation_level)
    }
}

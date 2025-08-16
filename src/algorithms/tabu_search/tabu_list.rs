use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};

use crate::utils::config::{ListType, TabuConf};
use crate::utils::opt_prob::FloatNumber as FloatNum;

#[derive(Debug, Clone, PartialEq)]
pub enum TabuType {
    Standard,
    Reactive {
        min_size: usize,
        max_size: usize,
        increase_factor: f64,
        decrease_factor: f64,
    },
    FrequencyBased {
        frequency_threshold: usize,
        max_frequency: usize,
    },
    QualityBased {
        quality_threshold: f64,
        quality_memory_size: usize,
    },
}

impl From<&TabuConf> for TabuType {
    fn from(conf: &TabuConf) -> Self {
        match &conf.list_type {
            ListType::Reactive(conf) => TabuType::Reactive {
                min_size: conf.min_tabu_size,
                max_size: conf.max_tabu_size,
                increase_factor: conf.increase_factor,
                decrease_factor: conf.decrease_factor,
            },
            ListType::FrequencyBased(conf) => TabuType::FrequencyBased {
                frequency_threshold: conf.frequency_threshold,
                max_frequency: conf.max_frequency,
            },
            ListType::QualityBased(conf) => TabuType::QualityBased {
                quality_threshold: conf.quality_threshold,
                quality_memory_size: conf.quality_memory_size,
            },
            ListType::Standard(_) => TabuType::Standard,
        }
    }
}

pub struct TabuList<T, D>
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    items: VecDeque<OVector<T, D>>,
    max_size: usize,
    tabu_type: TabuType,
    frequency_map: HashMap<String, usize>,
    quality_memory: VecDeque<(OVector<T, D>, T)>,
    best_fitness: Option<T>,
}

impl<T, D> TabuList<T, D>
where
    T: FloatNum,
    D: Dim,
    OVector<T, D>: Send + Sync,
    DefaultAllocator: Allocator<D>,
{
    pub fn new(max_size: usize, tabu_type: TabuType) -> Self {
        Self {
            items: VecDeque::with_capacity(max_size),
            max_size,
            tabu_type,
            frequency_map: HashMap::new(),
            quality_memory: VecDeque::new(),
            best_fitness: None,
        }
    }

    pub fn is_tabu(&self, x: &OVector<T, D>, threshold: T) -> bool {
        let in_tabu_list = self.items.par_iter().any(|tabu_x| {
            let diff = x - tabu_x;
            diff.dot(&diff).sqrt() < threshold
        });

        // Aspiration criteria- allow tabu move if it leads to the best solution found
        if in_tabu_list {
            if let Some(_best_f) = self.best_fitness {
                return false;
            }
            return true;
        }

        // Frequency-based restrictions
        if let TabuType::FrequencyBased {
            frequency_threshold,
            ..
        } = &self.tabu_type
        {
            let key = self.solution_key(x);
            if let Some(&freq) = self.frequency_map.get(&key) {
                if freq >= *frequency_threshold {
                    return true;
                }
            }
        }

        false
    }

    pub fn update(
        &mut self,
        x: OVector<T, D>,
        iterations_since_improvement: usize,
        fitness: Option<T>,
    ) {
        if let Some(f) = fitness {
            if self.best_fitness.is_none() || f > self.best_fitness.unwrap() {
                self.best_fitness = Some(f);
            }
        }

        match self.tabu_type {
            TabuType::Reactive {
                min_size,
                max_size,
                increase_factor,
                decrease_factor,
            } => {
                let new_size = if iterations_since_improvement > 10 {
                    ((self.items.len() as f64) * decrease_factor) as usize
                } else {
                    ((self.items.len() as f64) * increase_factor) as usize
                };
                self.max_size = new_size.clamp(min_size, max_size);
            }
            TabuType::FrequencyBased { max_frequency, .. } => {
                let key = self.solution_key(&x);
                let freq = self.frequency_map.entry(key).or_insert(0);
                *freq = (*freq + 1).min(max_frequency);
            }
            TabuType::QualityBased {
                quality_memory_size,
                ..
            } => {
                if let Some(f) = fitness {
                    self.quality_memory.push_front((x.clone(), f));
                    while self.quality_memory.len() > quality_memory_size {
                        self.quality_memory.pop_back();
                    }
                }
            }
            TabuType::Standard => {}
        }

        self.items.push_front(x);
        while self.items.len() > self.max_size {
            self.items.pop_back();
        }
    }

    pub fn can_aspire(&self, _x: &OVector<T, D>, current_fitness: T) -> bool {
        if let Some(best_f) = self.best_fitness {
            current_fitness > best_f
        } else {
            false
        }
    }

    pub fn get_frequency(&self, x: &OVector<T, D>) -> usize {
        let key = self.solution_key(x);
        *self.frequency_map.get(&key).unwrap_or(&0)
    }

    pub fn get_quality_score(&self, x: &OVector<T, D>) -> Option<T> {
        self.quality_memory
            .iter()
            .find(|(stored_x, _)| {
                let diff = x - stored_x;
                diff.dot(&diff).sqrt() < T::from_f64(1e-6).unwrap()
            })
            .map(|(_, fitness)| *fitness)
    }

    fn solution_key(&self, x: &OVector<T, D>) -> String {
        let precision = 6;
        let rounded: Vec<String> = x
            .iter()
            .map(|&val| format!("{:.prec$}", val.to_f64().unwrap(), prec = precision))
            .collect();
        rounded.join(",")
    }

    pub fn clear_frequency_map(&mut self) {
        self.frequency_map.clear();
    }

    pub fn reset_quality_memory(&mut self) {
        self.quality_memory.clear();
    }
}

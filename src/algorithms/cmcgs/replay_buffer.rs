use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rayon::prelude::*;

use crate::utils::opt_prob::FloatNumber as FloatNum;

#[derive(Clone, Debug)]
pub struct ExperienceTuple<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub state: OVector<T, D>,
    pub action: OVector<T, D>,
    pub next_state: OVector<T, D>,
    pub return_value: T,
    pub node_id: usize,
    pub timestep: usize,
}

pub struct ReplayBuffer<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    experiences: Vec<ExperienceTuple<T, D>>,
    max_size: usize,
}

impl<T, D> ReplayBuffer<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    pub fn new(max_size: usize) -> Self {
        Self {
            experiences: Vec::new(),
            max_size,
        }
    }

    pub fn add_experience(&mut self, experience: ExperienceTuple<T, D>) {
        self.experiences.push(experience);

        if self.experiences.len() > self.max_size {
            self.experiences.remove(0);
        }
    }

    pub fn get_experiences(&self) -> &[ExperienceTuple<T, D>] {
        &self.experiences
    }

    pub fn count_transitions_at_timestep(&self, timestep: usize) -> usize {
        self.experiences
            .par_iter()
            .filter(|e| e.timestep == timestep)
            .count()
    }

    pub fn get_states_at_timestep(&self, timestep: usize) -> Vec<OVector<T, D>> {
        self.experiences
            .par_iter()
            .filter(|e| e.timestep == timestep)
            .map(|e| e.state.clone())
            .collect()
    }

    pub fn clear(&mut self) {
        self.experiences.clear();
    }

    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }
}

pub struct NodeReplayBuffer<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    experiences: Vec<ExperienceTuple<T, D>>,
    max_size: usize,
}

impl<T, D> NodeReplayBuffer<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    pub fn new(max_size: usize) -> Self {
        Self {
            experiences: Vec::new(),
            max_size,
        }
    }

    pub fn add_experience(&mut self, experience: ExperienceTuple<T, D>) {
        self.experiences.push(experience);

        if self.experiences.len() > self.max_size {
            self.experiences.remove(0);
        }
    }

    pub fn get_experiences(&self) -> &[ExperienceTuple<T, D>] {
        &self.experiences
    }

    /// Elitism by return value
    pub fn get_top_experiences(&self, count: usize) -> Vec<&ExperienceTuple<T, D>> {
        let mut sorted: Vec<_> = self.experiences.iter().collect();
        sorted.sort_by(|a, b| {
            b.return_value
                .partial_cmp(&a.return_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(count).collect()
    }

    pub fn has_sufficient_experience(&self, threshold: usize) -> bool {
        self.experiences.len() > threshold / 2
    }

    pub fn clear(&mut self) {
        self.experiences.clear();
    }

    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }
}

pub struct NodeReplayBufferManager<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    buffers: std::collections::HashMap<usize, NodeReplayBuffer<T, D>>,
    max_size: usize,
}

impl<T, D> NodeReplayBufferManager<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    pub fn new(max_size: usize) -> Self {
        Self {
            buffers: std::collections::HashMap::new(),
            max_size,
        }
    }

    pub fn get_or_create_buffer(&mut self, node_id: usize) -> &mut NodeReplayBuffer<T, D> {
        self.buffers
            .entry(node_id)
            .or_insert_with(|| NodeReplayBuffer::new(self.max_size))
    }

    pub fn get_buffer(&self, node_id: usize) -> Option<&NodeReplayBuffer<T, D>> {
        self.buffers.get(&node_id)
    }

    pub fn get_buffer_mut(&mut self, node_id: usize) -> Option<&mut NodeReplayBuffer<T, D>> {
        self.buffers.get_mut(&node_id)
    }

    pub fn add_experience(&mut self, node_id: usize, experience: ExperienceTuple<T, D>) {
        let buffer = self.get_or_create_buffer(node_id);
        buffer.add_experience(experience);
    }

    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    pub fn get_node_ids(&self) -> Vec<usize> {
        self.buffers.keys().copied().collect()
    }
}

use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rand::prelude::*;

use super::gaussian_distributions::{GaussianActionPolicy, GaussianStateDistribution};
use crate::utils::opt_prob::FloatNumber as FloatNum;

#[derive(Clone, Debug)]
pub struct CMCGSGraphNode<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub id: usize,
    pub depth: usize,
    pub state_distribution: GaussianStateDistribution<T, D>,
    pub action_policy: GaussianActionPolicy<T, D>,
    pub visits: usize,
    pub total_reward: T,
    pub best_reward: T,
    pub parents: Vec<usize>,
    pub children: Vec<usize>,
}

impl<T, D> CMCGSGraphNode<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    pub fn new(id: usize, depth: usize, dimension: usize, action_bounds: (T, T)) -> Self {
        Self {
            id,
            depth,
            state_distribution: GaussianStateDistribution::new(dimension),
            action_policy: GaussianActionPolicy::new(dimension, action_bounds),
            visits: 0,
            total_reward: T::zero(),
            best_reward: T::neg_infinity(),
            parents: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn new_root(id: usize, initial_state: OVector<T, D>, action_bounds: (T, T)) -> Self {
        let dimension = initial_state.len();
        let mut node = Self::new(id, 0, dimension, action_bounds);
        node.state_distribution = GaussianStateDistribution::from_states(&[initial_state]);
        node
    }

    pub fn update(&mut self, reward: T) {
        self.visits += 1;
        self.total_reward += reward;

        if reward > self.best_reward {
            self.best_reward = reward;
        }
    }

    pub fn average_reward(&self) -> T {
        if self.visits == 0 {
            T::zero()
        } else {
            self.total_reward / T::from_usize(self.visits).unwrap()
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn is_root(&self) -> bool {
        self.parents.is_empty()
    }

    pub fn add_parent(&mut self, parent_id: usize) {
        if !self.parents.contains(&parent_id) {
            self.parents.push(parent_id);
        }
    }

    pub fn add_child(&mut self, child_id: usize) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
        }
    }

    pub fn get_state_probability(&self, state: &OVector<T, D>) -> T {
        self.state_distribution.probability_density(state)
    }

    pub fn update_state_distribution(&mut self, states: &[OVector<T, D>]) {
        self.state_distribution.update_from_states(states);
    }

    pub fn update_action_policy(&mut self, actions: &[OVector<T, D>], returns: &[T]) {
        self.action_policy
            .update_from_elite_experiences(actions, returns);
    }

    pub fn sample_action(&self, rng: &mut impl Rng) -> OVector<T, D> {
        self.action_policy.sample_action(rng)
    }

    pub fn get_state_distribution(&self) -> &GaussianStateDistribution<T, D> {
        &self.state_distribution
    }

    pub fn get_action_policy(&self) -> &GaussianActionPolicy<T, D> {
        &self.action_policy
    }

    pub fn get_state_distribution_mut(&mut self) -> &mut GaussianStateDistribution<T, D> {
        &mut self.state_distribution
    }

    pub fn get_action_policy_mut(&mut self) -> &mut GaussianActionPolicy<T, D> {
        &mut self.action_policy
    }
}

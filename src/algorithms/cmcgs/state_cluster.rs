use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rayon::prelude::*;

use crate::utils::opt_prob::FloatNumber as FloatNum;

#[derive(Clone, Debug)]
pub struct StateCluster<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub centroid: OVector<T, D>, // Representative state (centroid of cluster)
    pub states: Vec<OVector<T, D>>, // Similar states
    pub radius: T,
    pub size: usize,
    pub quality: T,
    dirty: bool, // Cache-like dirty bool: indicates need recalculation
}

impl<T, D> StateCluster<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    pub fn new(centroid: OVector<T, D>) -> Self {
        Self {
            centroid: centroid.clone(),
            states: vec![centroid],
            radius: T::from_f64(0.1).unwrap(),
            size: 1,
            quality: T::zero(),
            dirty: false,
        }
    }

    pub fn contains(&self, state: &OVector<T, D>) -> bool {
        let distance = self.distance_to_centroid(state);
        distance <= self.radius
    }

    pub fn distance_to_centroid(&self, state: &OVector<T, D>) -> T {
        let diff = state - &self.centroid;
        diff.dot(&diff).sqrt()
    }

    pub fn add_state(&mut self, state: OVector<T, D>) {
        self.states.push(state);
        self.size = self.states.len();
        self.dirty = true;
    }

    pub fn add_state_with_reward(&mut self, state: OVector<T, D>, reward: T) {
        self.add_state(state);

        // Update quality (exponential moving average)
        let alpha = T::from_f64(0.1).unwrap();
        self.quality = alpha * reward + (T::one() - alpha) * self.quality;
    }

    pub fn centroid(&self) -> &OVector<T, D> {
        &self.centroid
    }

    pub fn get_centroid(&self) -> OVector<T, D> {
        if self.dirty {
            let mut new_centroid =
                OVector::<T, D>::zeros_generic(D::from_usize(self.centroid.len()), nalgebra::U1);

            for state in &self.states {
                new_centroid += state;
            }
            new_centroid /= T::from_usize(self.size).unwrap();
            new_centroid
        } else {
            self.centroid.clone()
        }
    }

    pub fn can_merge_with(&self, other: &Self, merge_threshold: T) -> bool {
        let distance = self.distance_to_centroid(&other.centroid);
        distance <= merge_threshold
    }

    pub fn merge_with(&mut self, other: &Self) {
        self.states.extend(other.states.clone());
        self.size = self.states.len();
        self.dirty = true;

        // Update quality (weighted average)
        let total_quality = self.quality + other.quality;
        self.quality = total_quality / T::from_f64(2.0).unwrap();
    }
}

pub struct StateClusterManager<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub clusters: Vec<StateCluster<T, D>>,
    pub merge_threshold: T,
    pub max_clusters: usize,
}

impl<T, D> StateClusterManager<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    pub fn new() -> Self {
        Self {
            clusters: Vec::new(),
            merge_threshold: T::from_f64(0.5).unwrap(),
            max_clusters: 20,
        }
    }

    pub fn add_state(&mut self, state: OVector<T, D>) {
        let closest_cluster = self
            .clusters
            .par_iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a = a.distance_to_centroid(&state);
                let dist_b = b.distance_to_centroid(&state);
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        if let Some(cluster_idx) = closest_cluster {
            if self.clusters[cluster_idx].contains(&state) {
                self.clusters[cluster_idx].add_state(state);
            } else {
                self.create_new_cluster(state);
            }
        } else {
            self.create_new_cluster(state);
        }

        self.merge_clusters_if_needed();
    }

    pub fn add_state_with_reward(&mut self, state: OVector<T, D>, reward: T) {
        let closest_cluster = self
            .clusters
            .par_iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a = a.distance_to_centroid(&state);
                let dist_b = b.distance_to_centroid(&state);
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        if let Some(cluster_idx) = closest_cluster {
            if self.clusters[cluster_idx].contains(&state) {
                self.clusters[cluster_idx].add_state_with_reward(state, reward);
            } else {
                self.create_new_cluster_with_reward(state, reward);
            }
        } else {
            self.create_new_cluster_with_reward(state, reward);
        }

        self.merge_clusters_if_needed();
    }

    fn create_new_cluster(&mut self, state: OVector<T, D>) {
        let cluster = StateCluster::new(state);
        self.clusters.push(cluster);
    }

    fn create_new_cluster_with_reward(&mut self, state: OVector<T, D>, reward: T) {
        let mut cluster = StateCluster::new(state.clone());
        cluster.add_state_with_reward(state, reward);
        self.clusters.push(cluster);
    }

    /// Cluster states using simple agglomerative clustering (merge closest clusters)
    pub fn cluster_states_in_layer(
        &mut self,
        states: Vec<&OVector<T, D>>,
        desired_clusters: usize,
    ) -> Vec<StateCluster<T, D>> {
        if states.is_empty() {
            return Vec::new();
        }

        let desired_clusters = desired_clusters.min(states.len()).min(self.max_clusters);

        if desired_clusters <= 1 {
            let states_vec: Vec<_> = states.into_iter().cloned().collect();
            if let Some(first_state) = states_vec.first() {
                return vec![StateCluster::new(first_state.clone())];
            } else {
                return Vec::new();
            }
        }

        self.ward_linkage_clustering(states, desired_clusters)
    }

    pub fn clear(&mut self) {
        self.clusters.clear();
    }

    pub fn get_best_cluster(&self) -> Option<&StateCluster<T, D>> {
        self.clusters.iter().max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    // If clusters are too close, merge them
    fn merge_clusters_if_needed(&mut self) {
        let mut i = 0;
        while i < self.clusters.len() {
            let mut j = i + 1;
            while j < self.clusters.len() {
                if self.clusters[i].can_merge_with(&self.clusters[j], self.merge_threshold) {
                    let cluster_j = self.clusters.remove(j);
                    self.clusters[i].merge_with(&cluster_j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    // Use Ward linkage agglomerative clustering
    fn ward_linkage_clustering(
        &self,
        states: Vec<&OVector<T, D>>,
        k: usize,
    ) -> Vec<StateCluster<T, D>> {
        if states.is_empty() || k == 0 {
            return Vec::new();
        }

        let n_states = states.len();
        if n_states <= k {
            return states
                .into_iter()
                .map(|state| StateCluster::new(state.clone()))
                .collect();
        }

        let mut clusters: Vec<OVector<T, D>> = states.into_iter().cloned().collect();

        // Merge clusters until we have k clusters
        while clusters.len() > k {
            let (best_i, best_j) = self.find_best_ward_merge(&clusters);
            let centroid_i = clusters[best_i].clone();
            let centroid_j = clusters.remove(best_j);
            clusters[best_i] = (centroid_i + centroid_j) / T::from_f64(2.0).unwrap();
        }

        clusters
            .into_iter()
            .map(|centroid| StateCluster::new(centroid))
            .collect()
    }

    fn find_best_ward_merge(&self, clusters: &[OVector<T, D>]) -> (usize, usize) {
        let n_clusters = clusters.len();
        let mut best_merge = (0, 1);
        let mut best_increase = T::infinity();

        for i in 0..n_clusters {
            for j in (i + 1)..n_clusters {
                let increase = self.ward_linkage_increase(&clusters[i], &clusters[j]);
                if increase < best_increase {
                    best_increase = increase;
                    best_merge = (i, j);
                }
            }
        }

        best_merge
    }

    // Minimise increase in within-cluster sum of squares
    fn ward_linkage_increase(&self, cluster_a: &OVector<T, D>, cluster_b: &OVector<T, D>) -> T {
        let centroid_combined = (cluster_a + cluster_b) / T::from_f64(2.0).unwrap();
        let diff_a = cluster_a - &centroid_combined;
        let diff_b = cluster_b - &centroid_combined;
        diff_a.dot(&diff_a) + diff_b.dot(&diff_b)
    }
}

impl<T, D> Default for StateClusterManager<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

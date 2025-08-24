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

        // Update centroid (simple average)
        let mut new_centroid =
            OVector::<T, D>::zeros_generic(D::from_usize(self.centroid.len()), nalgebra::U1);

        for state in &self.states {
            new_centroid += state;
        }
        new_centroid /= T::from_usize(self.size).unwrap();

        self.centroid = new_centroid;

        // Update radius to encompass all states
        let mut max_distance = T::zero();
        for state in &self.states {
            let distance = self.distance_to_centroid(state);
            if distance > max_distance {
                max_distance = distance;
            }
        }
        self.radius = max_distance;
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

    pub fn can_merge_with(&self, other: &Self, merge_threshold: T) -> bool {
        let distance = self.distance_to_centroid(&other.centroid);
        distance <= merge_threshold
    }

    pub fn merge_with(&mut self, other: &Self) {
        self.states.extend(other.states.clone());
        self.size = self.states.len();

        let mut new_centroid =
            OVector::<T, D>::zeros_generic(D::from_usize(self.centroid.len()), nalgebra::U1);

        for state in &self.states {
            new_centroid += state;
        }
        new_centroid /= T::from_usize(self.size).unwrap();

        self.centroid = new_centroid;

        let mut max_distance = T::zero();
        for state in &self.states {
            let distance = self.distance_to_centroid(state);
            if distance > max_distance {
                max_distance = distance;
            }
        }
        self.radius = max_distance;

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

        let mut new_clusters = Vec::new();
        for &state in &states {
            let cluster = StateCluster::new(state.clone());
            new_clusters.push(cluster);
        }

        // Merge clusters until we have the desired number
        while new_clusters.len() > desired_clusters {
            let pairs: Vec<_> = (0..new_clusters.len())
                .flat_map(|i| (i + 1..new_clusters.len()).map(move |j| (i, j)))
                .collect();

            let best_pair = pairs.par_iter().min_by(|&&(i1, j1), &&(i2, j2)| {
                let dist1 = new_clusters[i1].distance_to_centroid(&new_clusters[j1].centroid);
                let dist2 = new_clusters[i2].distance_to_centroid(&new_clusters[j2].centroid);
                dist1
                    .partial_cmp(&dist2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Merge clusters i and j, keeping the one with lower index
            if let Some(&(i, j)) = best_pair {
                let other_cluster = new_clusters.remove(j);
                new_clusters[i].merge_with(&other_cluster);
            } else {
                break;
            }
        }

        // Must contain at least m/2 states
        new_clusters.retain(|cluster| cluster.size >= states.len() / 2);

        new_clusters
    }

    fn merge_clusters_if_needed(&mut self) {
        while self.clusters.len() > self.max_clusters {
            // Parallel search for best pair to merge
            let pairs: Vec<_> = (0..self.clusters.len())
                .flat_map(|i| (i + 1..self.clusters.len()).map(move |j| (i, j)))
                .collect();

            let best_pair = pairs.par_iter().min_by(|&&(i1, j1), &&(i2, j2)| {
                let dist1 = self.clusters[i1].distance_to_centroid(&self.clusters[j1].centroid);
                let dist2 = self.clusters[i2].distance_to_centroid(&self.clusters[j2].centroid);
                dist1
                    .partial_cmp(&dist2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(&(i, j)) = best_pair {
                let other_cluster = self.clusters.remove(j);
                self.clusters[i].merge_with(&other_cluster);
            } else {
                break;
            }
        }
    }

    pub fn get_best_cluster(&self) -> Option<&StateCluster<T, D>> {
        self.clusters.par_iter().max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn get_clusters_by_quality(&self) -> Vec<&StateCluster<T, D>> {
        let mut clusters: Vec<_> = self.clusters.iter().collect();
        clusters.par_sort_by(|a, b| {
            b.quality
                .partial_cmp(&a.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        clusters
    }

    pub fn clear(&mut self) {
        self.clusters.clear();
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

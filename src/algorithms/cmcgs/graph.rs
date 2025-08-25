use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector};
use rayon::prelude::*;

use crate::algorithms::cmcgs::{graph_node::CMCGSGraphNode, state_cluster::StateCluster};
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub struct CMCGSGraph<T, D>
where
    T: FloatNum,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub nodes: Vec<CMCGSGraphNode<T, D>>,
    pub roots: Vec<usize>,
    next_node_id: usize,
}

impl<T, D> CMCGSGraph<T, D>
where
    T: FloatNum + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D>,
    OVector<T, D>: Send + Sync,
{
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
            next_node_id: 0,
        }
    }

    pub fn create_root_node(&mut self, initial_state: OVector<T, D>) -> usize {
        let action_bounds = (T::from_f64(-1.0).unwrap(), T::from_f64(1.0).unwrap());
        let root_node = CMCGSGraphNode::new_root(self.next_node_id, initial_state, action_bounds);
        let node_id = self.next_node_id;

        self.nodes.push(root_node);
        self.roots.push(node_id);
        self.next_node_id += 1;

        node_id
    }

    pub fn add_placeholder_node(
        &mut self,
        depth: usize,
        placeholder_state: OVector<T, D>,
    ) -> usize {
        let action_bounds = (T::from_f64(-1.0).unwrap(), T::from_f64(1.0).unwrap());
        let node = CMCGSGraphNode::new(
            self.next_node_id,
            depth,
            placeholder_state.len(),
            action_bounds,
        );
        let node_id = self.next_node_id;

        self.nodes.push(node);
        self.next_node_id += 1;

        node_id
    }

    pub fn add_node(&mut self, node: CMCGSGraphNode<T, D>) -> usize {
        let node_id = node.id;
        self.nodes.push(node);
        node_id
    }

    pub fn connect_nodes(&mut self, parent_id: usize, child_id: usize) {
        if let Some(parent) = self.get_node_mut(parent_id) {
            parent.add_child(child_id);
        }

        if let Some(child) = self.get_node_mut(child_id) {
            child.add_parent(parent_id);
        }
    }

    pub fn get_node(&self, node_id: usize) -> Option<&CMCGSGraphNode<T, D>> {
        self.nodes.iter().find(|node| node.id == node_id)
    }

    pub fn get_node_mut(&mut self, node_id: usize) -> Option<&mut CMCGSGraphNode<T, D>> {
        self.nodes.iter_mut().find(|node| node.id == node_id)
    }

    pub fn get_root_id(&self) -> usize {
        self.roots.first().copied().unwrap_or(0)
    }

    pub fn get_nodes_at_depth(&self, depth: usize) -> Vec<usize> {
        self.nodes
            .par_iter()
            .enumerate()
            .filter(|(_, node)| node.depth == depth)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn get_layer_size(&self, depth: usize) -> usize {
        self.get_nodes_at_depth(depth).len()
    }

    pub fn get_max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0) + 1
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_visits(&self) -> usize {
        self.nodes.par_iter().map(|n| n.visits).sum()
    }

    pub fn replace_layer_nodes(&mut self, depth: usize, new_clusters: Vec<StateCluster<T, D>>) {
        self.nodes.retain(|node| node.depth != depth);

        if depth == 0 {
            self.roots.clear();
        }

        for cluster in new_clusters.into_iter() {
            let action_bounds = (T::from_f64(-1.0).unwrap(), T::from_f64(1.0).unwrap());
            let centroid = cluster.centroid().clone();
            let mut new_node =
                CMCGSGraphNode::new(self.next_node_id, depth, centroid.len(), action_bounds);

            new_node.update_state_distribution(&[centroid]);

            if depth == 0 {
                self.roots.push(self.next_node_id);
            }

            self.nodes.push(new_node);
            self.next_node_id += 1;
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.roots.clear();
        self.next_node_id = 0;
    }
}

impl<T, D> Default for CMCGSGraph<T, D>
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

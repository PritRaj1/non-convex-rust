use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, RealField, U1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::utils::config::CMCGSConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, OptimizationAlgorithm, State};

use crate::algorithms::cmcgs::{
    graph::CMCGSGraph,
    replay_buffer::{ExperienceTuple, NodeReplayBufferManager, ReplayBuffer},
    state_cluster::StateClusterManager,
};

type TrajectoryEntry<T, D> = (OVector<T, D>, OVector<T, D>, OVector<T, D>, usize);

pub struct CMCGS<T, N, D>
where
    T: FloatNum + RealField + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: CMCGSConf,
    pub opt_prob: OptProb<T, D>,
    pub st: State<T, N, D>,
    pub replay_buffer: ReplayBuffer<T, D>,
    pub node_replay_buffers: NodeReplayBufferManager<T, D>,
    pub state_cluster_manager: StateClusterManager<T, D>,
    pub graph: CMCGSGraph<T, D>,
    pub iteration_count: usize,
    pub last_improvement: usize,
    pub stagnation_count: usize,
    pub restart_count: usize,
    pub rng: StdRng,
    pub seed: u64,
    pub cached_lower_bounds: Option<OVector<T, D>>,
    pub cached_upper_bounds: Option<OVector<T, D>>,
}

impl<T, N, D> CMCGS<T, N, D>
where
    T: FloatNum + RealField + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub fn new(
        conf: CMCGSConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        seed: u64,
    ) -> Self {
        let init_x: OVector<T, D> = init_pop.row(0).transpose().into_owned();
        let best_f = opt_prob.evaluate(&init_x);
        let n = init_x.len();

        let cached_lower_bounds = opt_prob.objective.x_lower_bound(&init_x);
        let cached_upper_bounds = opt_prob.objective.x_upper_bound(&init_x);

        let replay_buffer = ReplayBuffer::new(10000);
        let node_replay_buffers = NodeReplayBufferManager::new(1000);
        let state_cluster_manager = StateClusterManager::new();
        let graph = CMCGSGraph::new();

        Self {
            conf,
            opt_prob: opt_prob.clone(),
            st: State {
                best_x: init_x.clone(),
                best_f,
                pop: OMatrix::<T, N, D>::from_fn_generic(
                    N::from_usize(1),
                    D::from_usize(n),
                    |_, j| init_x.clone()[j],
                ),
                fitness: OVector::<T, N>::from_element_generic(N::from_usize(1), U1, best_f),
                constraints: OVector::<bool, N>::from_element_generic(
                    N::from_usize(1),
                    U1,
                    opt_prob.is_feasible(&init_x.clone()),
                ),
                iter: 1,
            },
            replay_buffer,
            node_replay_buffers,
            state_cluster_manager,
            graph,
            iteration_count: 0,
            last_improvement: 0,
            stagnation_count: 0,
            restart_count: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
            cached_lower_bounds,
            cached_upper_bounds,
        }
    }

    fn initialize_graph(&mut self) {
        let root_node_id = self.graph.create_root_node(self.st.best_x.clone());

        // Initialize graph with d_init layers, each containing one node initially
        for layer in 1..self.conf.max_depth {
            let placeholder_state = self.generate_random_state();
            let node_id = self.graph.add_placeholder_node(layer, placeholder_state);
            self.node_replay_buffers.get_or_create_buffer(node_id);
        }

        self.node_replay_buffers.get_or_create_buffer(root_node_id);
    }

    fn perform_cmcgs_iteration(&mut self) -> T {
        let (trajectory, final_state, _final_node) = self.selection_phase();
        self.depth_expansion_phase();
        let rollout_return = self.simulation_phase(&final_state);
        self.backup_phase(&trajectory, rollout_return);
        self.width_expansion_phase();
        rollout_return
    }

    // Traverse graph with epsilon-greedy to select next node
    fn selection_phase(&mut self) -> (Vec<TrajectoryEntry<T, D>>, OVector<T, D>, usize) {
        let mut trajectory = Vec::new();
        let mut current_state = self.st.best_x.clone();
        let mut current_node_id = self.graph.get_root_id();
        let mut depth = 0;

        while !self.is_terminal_state(&current_state) && depth < self.conf.max_depth {
            let action = if self.rng.random::<f64>() < self.conf.epsilon {
                self.sample_from_node_policy(current_node_id)
            } else {
                self.sample_greedy_action_with_noise(current_node_id)
            };

            let (next_state, _reward) = self.apply_dynamics_model(&current_state, &action);

            trajectory.push((
                current_state.clone(),
                action,
                next_state.clone(),
                current_node_id,
            ));

            if self.is_terminal_state(&next_state) || depth == (self.conf.max_depth - 1) {
                break;
            }

            current_node_id = self.select_next_node(depth + 1, &next_state);
            current_state = next_state;
            depth += 1;
        }

        (trajectory, current_state, current_node_id)
    }

    /// Sample action from node policy π_q(a) - Gaussian distribution
    fn sample_from_node_policy(&mut self, node_id: usize) -> OVector<T, D> {
        if let Some(node) = self.graph.get_node(node_id) {
            return node.sample_action(&mut self.rng.clone());
        }

        self.generate_random_action() // Fallback: random action
    }

    // Take top action and add Gaussian noise for local improvement
    fn sample_greedy_action_with_noise(&mut self, node_id: usize) -> OVector<T, D> {
        if let Some(buffer) = self.node_replay_buffers.get_buffer(node_id) {
            if !buffer.is_empty() {
                let top_experiences = buffer.get_top_experiences(1);
                if let Some(top_exp) = top_experiences.first() {
                    let top_action = &top_exp.action;
                    let noise_std = T::from_f64(0.1).unwrap(); // E_top from paper

                    let mut noisy_action = top_action.clone();
                    for i in 0..noisy_action.len() {
                        let noise =
                            T::from_f64(self.rng.random_range(-2.0..2.0)).unwrap() * noise_std;
                        noisy_action[i] += noise;
                    }

                    return noisy_action;
                }
            }
        }

        self.generate_random_action() // Fallback: random action
    }

    fn select_next_node(&self, layer: usize, state: &OVector<T, D>) -> usize {
        let layer_nodes = self.graph.get_nodes_at_depth(layer);

        if let Some(best_node_idx) = layer_nodes.par_iter().max_by(|&&a, &&b| {
            let node_a = &self.graph.get_node(a).unwrap();
            let node_b = &self.graph.get_node(b).unwrap();

            let prob_a = node_a.get_state_probability(state);
            let prob_b = node_b.get_state_probability(state);

            prob_a
                .partial_cmp(&prob_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            *best_node_idx
        } else {
            layer_nodes.first().copied().unwrap_or(0) // Fallback: first node in layer
        }
    }

    // Add new layer if threshold m is met
    fn depth_expansion_phase(&mut self) {
        let current_depth = self.graph.get_max_depth();

        if current_depth < self.conf.max_depth {
            let last_layer_nodes = self.graph.get_nodes_at_depth(current_depth - 1);
            let total_experience: usize = last_layer_nodes
                .par_iter()
                .map(|&node_id| {
                    self.node_replay_buffers
                        .get_buffer(node_id)
                        .map(|buffer| buffer.len())
                        .unwrap_or(0)
                })
                .sum();

            if total_experience > self.conf.expansion_threshold {
                let new_depth = current_depth;
                let placeholder_state = self.generate_random_state();
                let new_node_id = self
                    .graph
                    .add_placeholder_node(new_depth, placeholder_state);

                self.node_replay_buffers.get_or_create_buffer(new_node_id);
            }
        }
    }

    // Use dynamics model to perform a random rollout, accumulate discounted rewards
    fn simulation_phase(&mut self, start_state: &OVector<T, D>) -> T {
        let mut current_state = start_state.clone();
        let mut total_return = T::zero();
        let rollout_steps = self.conf.simulation_steps;

        for step in 0..rollout_steps {
            if self.is_terminal_state(&current_state) {
                break;
            }

            // Random action for rollout
            let action = self.generate_random_action();
            let (next_state, reward) = self.apply_dynamics_model(&current_state, &action);

            // Apply discount factor γ
            total_return += num_traits::Float::powi(
                T::from_f64(self.conf.discount_factor).unwrap(),
                step as i32,
            ) * reward;
            current_state = next_state;
        }

        total_return
    }

    // Update node distributions and replay buffers
    fn backup_phase(&mut self, trajectory: &[TrajectoryEntry<T, D>], rollout_return: T) {
        for (state, action, next_state, node_id) in trajectory {
            let experience = ExperienceTuple {
                state: state.clone(),
                action: action.clone(),
                next_state: next_state.clone(),
                return_value: rollout_return,
                node_id: *node_id,
            };

            self.replay_buffer.add_experience(experience.clone());
            self.node_replay_buffers
                .add_experience(*node_id, experience);

            let has_sufficient = self
                .node_replay_buffers
                .get_buffer(*node_id)
                .map(|buffer| buffer.has_sufficient_experience(self.conf.expansion_threshold))
                .unwrap_or(false);

            if has_sufficient {
                self.update_node_distributions(*node_id);
            }
        }
    }

    // Update state and action distributions, p_q(s) and π_q(a)
    fn update_node_distributions(&mut self, node_id: usize) {
        if let Some(node) = self.graph.get_node_mut(node_id) {
            if let Some(buffer) = self.node_replay_buffers.get_buffer(node_id) {
                let states: Vec<_> = buffer
                    .get_experiences()
                    .iter()
                    .map(|e| e.state.clone())
                    .collect();
                node.update_state_distribution(&states);

                let actions: Vec<_> = buffer
                    .get_experiences()
                    .iter()
                    .map(|e| e.action.clone())
                    .collect();
                let returns: Vec<_> = buffer
                    .get_experiences()
                    .iter()
                    .map(|e| e.return_value)
                    .collect();
                node.update_action_policy(&actions, &returns);
            }
        }
    }

    // Cluster states and add new nodes if less than desired min(n_max, ⌊n_t/m⌋)
    fn width_expansion_phase(&mut self) {
        for layer in 0..self.graph.get_max_depth() {
            let transitions_at_timestep = self.replay_buffer.count_transitions_at_timestep(layer);
            let desired_nodes = (transitions_at_timestep / self.conf.expansion_threshold)
                .min(self.conf.max_nodes_per_layer);

            let current_nodes = self.graph.get_layer_size(layer);

            if current_nodes < desired_nodes {
                let new_clusters = self.cluster_states_in_layer(layer);

                if !new_clusters.is_empty() {
                    self.graph.replace_layer_nodes(layer, new_clusters);
                    self.reassign_experiences_to_nodes(layer);
                }
            }
        }
    }

    fn cluster_states_in_layer(
        &mut self,
        layer: usize,
    ) -> Vec<crate::algorithms::cmcgs::state_cluster::StateCluster<T, D>> {
        let layer_nodes = self.graph.get_nodes_at_depth(layer);
        let mut all_states = Vec::new();

        for &node_id in &layer_nodes {
            if let Some(buffer) = self.node_replay_buffers.get_buffer(node_id) {
                for experience in buffer.get_experiences() {
                    all_states.push(experience.state.clone());
                }
            }
        }

        if all_states.len() < (self.conf.expansion_threshold / 2) {
            return Vec::new(); // Not enough data for clustering
        }

        // Simple agglomerative clustering
        self.state_cluster_manager.cluster_states_in_layer(
            all_states.iter().collect(),
            layer_nodes.len() + 1, // Try to add one more cluster
        )
    }

    fn reassign_experiences_to_nodes(&mut self, layer: usize) {
        let layer_nodes = self.graph.get_nodes_at_depth(layer);
        if layer_nodes.is_empty() {
            return;
        }

        // Match experience to node based on state similarity
        let layer_experiences: Vec<_> = self
            .replay_buffer
            .get_experiences()
            .par_iter()
            .filter_map(|e| {
                let closest_node = layer_nodes.par_iter().min_by(|&&a, &&b| {
                    let node_a = self.graph.get_node(a).unwrap();
                    let node_b = self.graph.get_node(b).unwrap();
                    let dist_a = node_a.get_state_probability(&e.state);
                    let dist_b = node_b.get_state_probability(&e.state);
                    dist_a
                        .partial_cmp(&dist_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?;

                Some((e.clone(), *closest_node))
            })
            .collect();

        // Batch reassign experiences to nodes
        for (experience, node_id) in layer_experiences {
            self.node_replay_buffers.add_experience(node_id, experience);
        }
    }

    // Check if state is infeasible
    fn is_terminal_state(&self, state: &OVector<T, D>) -> bool {
        if let (Some(lower), Some(upper)) = (&self.cached_lower_bounds, &self.cached_upper_bounds) {
            for i in 0..state.len() {
                if state[i] < lower[i] || state[i] > upper[i] {
                    return true;
                }
            }
        }
        false
    }

    fn apply_dynamics_model(
        &self,
        state: &OVector<T, D>,
        action: &OVector<T, D>,
    ) -> (OVector<T, D>, T) {
        let next_state = state + action; // s_{t+1} = s_t + a_t
        let reward = -next_state.norm(); // Negative distance for minimization (reward = -cost)
        (next_state, reward)
    }

    fn generate_random_state(&mut self) -> OVector<T, D> {
        let dim = D::try_to_usize().unwrap_or(2);
        let mut state = OVector::zeros_generic(D::from_usize(dim), U1::from_usize(1));

        for i in 0..dim {
            state[i] = T::from_f64(self.rng.random_range(-5.0..5.0)).unwrap();
        }

        state
    }

    fn generate_random_action(&mut self) -> OVector<T, D> {
        let dim = D::try_to_usize().unwrap_or(2);
        let mut action = OVector::zeros_generic(D::from_usize(dim), U1::from_usize(1));

        for i in 0..dim {
            action[i] = T::from_f64(self.rng.random_range(-1.0..1.0)).unwrap();
        }

        action
    }

    fn should_restart(&self) -> bool {
        self.stagnation_count > self.conf.restart_threshold
    }

    fn restart(&mut self) {
        self.restart_count += 1;
        self.stagnation_count = 0;

        let (lb, ub) = self.get_bounds(&self.st.best_x);
        let mut new_solution =
            OVector::<T, D>::zeros_generic(D::from_usize(self.st.best_x.len()), U1);

        let mut attempts = 0;
        let max_attempts = 100;

        while attempts < max_attempts {
            for i in 0..new_solution.len() {
                new_solution[i] = T::from_f64(
                    self.rng
                        .random_range(lb[i].to_f64().unwrap()..ub[i].to_f64().unwrap()),
                )
                .unwrap();
            }

            if self.opt_prob.is_feasible(&new_solution) {
                break;
            }
            attempts += 1;
        }

        self.st.best_x = new_solution.clone();
        self.st.best_f = self.opt_prob.evaluate(&new_solution);
        for i in 0..new_solution.len() {
            self.st.pop[(0, i)] = new_solution[i];
        }
        self.st.fitness[0] = self.st.best_f;
        self.st.constraints[0] = self.opt_prob.is_feasible(&new_solution);

        self.replay_buffer.clear();
        self.node_replay_buffers.clear();
        self.state_cluster_manager.clear();
        self.graph.clear();

        self.state_cluster_manager
            .add_state_with_reward(new_solution.clone(), self.st.fitness[0]);
    }

    fn get_bounds(&self, candidate: &OVector<T, D>) -> (OVector<T, D>, OVector<T, D>) {
        if let (Some(lb), Some(ub)) = (&self.cached_lower_bounds, &self.cached_upper_bounds) {
            (lb.clone(), ub.clone())
        } else {
            let lb = self
                .opt_prob
                .objective
                .x_lower_bound(candidate)
                .unwrap_or_else(|| {
                    OVector::<T, D>::from_element_generic(
                        D::from_usize(candidate.len()),
                        U1,
                        T::from_f64(-10.0).unwrap(),
                    )
                });
            let ub = self
                .opt_prob
                .objective
                .x_upper_bound(candidate)
                .unwrap_or_else(|| {
                    OVector::<T, D>::from_element_generic(
                        D::from_usize(candidate.len()),
                        U1,
                        T::from_f64(10.0).unwrap(),
                    )
                });
            (lb, ub)
        }
    }

    fn update_best_solution(&mut self) {
        if let Some(best_cluster) = self.state_cluster_manager.get_best_cluster() {
            let new_state = best_cluster.centroid.clone();

            if self.opt_prob.is_feasible(&new_state) {
                let fitness = self.opt_prob.evaluate(&new_state);

                if fitness > self.st.best_f {
                    self.st.best_f = fitness;
                    self.st.best_x = new_state.clone();
                }

                // Update population with current best solution
                for i in 0..new_state.len() {
                    self.st.pop[(0, i)] = new_state[i];
                }
                self.st.fitness[0] = fitness;
                self.st.constraints[0] = self.opt_prob.is_feasible(&new_state);
            }
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for CMCGS<T, N, D>
where
    T: FloatNum + RealField + Send + Sync,
    N: Dim + Send + Sync,
    D: Dim + Send + Sync,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OVector<bool, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    fn step(&mut self) {
        if self.should_restart() {
            self.restart();
            return;
        }

        if self.iteration_count == 0 {
            self.initialize_graph();
        }

        for _ in 0..self.conf.simulation_count {
            self.perform_cmcgs_iteration();
        }

        let old_best_f = self.st.best_f;
        self.update_best_solution();

        if self.st.best_f > old_best_f {
            self.last_improvement = self.iteration_count;
            self.stagnation_count = 0;
        } else {
            self.stagnation_count += 1;
        }

        // Ensure population is always updated with current best solution
        if self.st.best_f > old_best_f {
            // If we found a better solution, update population
            for i in 0..self.st.best_x.len() {
                self.st.pop[(0, i)] = self.st.best_x[i];
            }
            self.st.fitness[0] = self.st.best_f;
            self.st.constraints[0] = self.opt_prob.is_feasible(&self.st.best_x);
        }

        self.iteration_count += 1;
        self.st.iter = self.iteration_count;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }

    fn get_cmcgs_tree(&self) -> Option<&super::graph::CMCGSGraph<T, D>> {
        Some(&self.graph)
    }
}

use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use rayon::prelude::*;

/// Reorders mesh entities using the Cuthill-McKee algorithm.
/// This improves memory locality and is useful for solver optimization.
pub fn cuthill_mckee(
    entities: &[MeshEntity], 
    adjacency: &FxHashMap<MeshEntity, Vec<MeshEntity>>
) -> Vec<MeshEntity> {
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    let mut ordered = Vec::new();

    if let Some((start, _)) = entities.iter()
        .map(|entity| (entity, adjacency.get(entity).map_or(0, |neighbors| neighbors.len())))
        .min_by_key(|&(_, degree)| degree)
    {
        queue.push_back(*start);
        visited.insert(*start);
    }

    while let Some(entity) = queue.pop_front() {
        ordered.push(entity);
        if let Some(neighbors) = adjacency.get(&entity) {
            let mut sorted_neighbors: Vec<_> = neighbors.iter()
                .filter(|&&n| !visited.contains(&n))
                .cloned()
                .collect();
            sorted_neighbors.sort_by_key(|n| adjacency.get(n).map_or(0, |neighbors| neighbors.len()));
            for neighbor in sorted_neighbors {
                queue.push_back(neighbor);
                visited.insert(neighbor);
            }
        }
    }

    ordered
}


impl Mesh {

    pub fn apply_reordering(&mut self, new_order: &[usize]) {
        // Implement the application of reordering to mesh entities or sparse matrix structure.
    }
    
    pub fn rcm_ordering(&self, start_node: MeshEntity) -> Vec<MeshEntity> {
        let mut visited = FxHashSet::default();
        let mut queue = VecDeque::new();
        let mut ordering = Vec::new();

        queue.push_back(start_node);
        visited.insert(start_node);

        while let Some(node) = queue.pop_front() {
            ordering.push(node);
            if let Some(neighbors) = self.sieve.cone(&node) {
                let mut sorted_neighbors: Vec<_> = neighbors
                    .into_iter()
                    .filter(|n| !visited.contains(n))
                    .collect();
                sorted_neighbors.sort_by_key(|n| self.sieve.cone(n).map_or(0, |set| set.len()));
                for neighbor in sorted_neighbors {
                    queue.push_back(neighbor);
                    visited.insert(neighbor);
                }
            }
        }

        ordering.reverse();
        ordering
    }

    pub fn reorder_by_morton_order(&mut self, elements: &mut [(u32, u32)]) {
        elements.par_sort_by_key(|&(x, y)| Self::morton_order_2d(x, y));
    }

    pub fn morton_order_2d(x: u32, y: u32) -> u64 {
        fn part1by1(n: u32) -> u64 {
            let mut n = n as u64;
            n = (n | (n << 16)) & 0x0000_0000_ffff_0000;
            n = (n | (n << 8)) & 0x0000_ff00_00ff_0000;
            n = (n | (n << 4)) & 0x00f0_00f0_00f0_00f0;
            n = (n | (n << 2)) & 0x0c30_0c30_0c30_0c30;
            n = (n | (n << 1)) & 0x2222_2222_2222_2222;
            n
        }

        part1by1(x) | (part1by1(y) << 1)
    }
}

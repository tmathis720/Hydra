use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use rayon::prelude::*;

/// Reorders mesh entities using the Cuthill-McKee algorithm.  
/// This algorithm improves memory locality by reducing the bandwidth of sparse matrices,  
/// which is beneficial for solver optimizations.  
///
/// The algorithm starts from the node with the smallest degree and visits its neighbors  
/// in increasing order of their degree.
///
/// Example usage:
/// 
///    let ordered_entities = cuthill_mckee(&entities, &adjacency);  
///
pub fn cuthill_mckee(
    entities: &[MeshEntity], 
    adjacency: &FxHashMap<MeshEntity, Vec<MeshEntity>>
) -> Vec<MeshEntity> {
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    let mut ordered = Vec::new();

    // Find the starting entity (node) with the smallest degree.
    if let Some((start, _)) = entities.iter()
        .map(|entity| (entity, adjacency.get(entity).map_or(0, |neighbors| neighbors.len())))
        .min_by_key(|&(_, degree)| degree)
    {
        queue.push_back(*start);
        visited.insert(*start);
    }

    // Perform the Cuthill-McKee reordering.
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
    /// Applies a reordering to the mesh entities based on the given new order.  
    ///
    /// This method can be used to reorder entities or update a sparse matrix  
    /// structure based on the new ordering.
    ///
    /// Example usage:
    /// 
    ///    mesh.apply_reordering(&new_order);  
    ///
    pub fn apply_reordering(&mut self, _new_order: &[usize]) {
        // Implement the application of reordering to mesh entities or sparse matrix structure.
    }

    /// Computes the reverse Cuthill-McKee (RCM) ordering starting from a given node.  
    ///
    /// This method performs the RCM algorithm to minimize the bandwidth of sparse matrices  
    /// by reordering mesh entities in reverse order of their Cuthill-McKee ordering.  
    ///
    /// Example usage:
    /// 
    ///    let rcm_order = mesh.rcm_ordering(start_node);  
    ///
    pub fn rcm_ordering(&self, start_node: MeshEntity) -> Vec<MeshEntity> {
        let mut visited = FxHashSet::default();
        let mut queue = VecDeque::new();
        let mut ordering = Vec::new();

        queue.push_back(start_node);
        visited.insert(start_node);

        // Perform breadth-first traversal and order nodes by degree.
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

        // Reverse the ordering to get the RCM order.
        ordering.reverse();
        ordering
    }

    /// Reorders elements in the mesh using Morton order (Z-order curve) for better memory locality.  
    ///
    /// This method applies the Morton order to the given set of 2D elements (with x and y coordinates).  
    /// Morton ordering is a space-filling curve that helps improve memory access patterns  
    /// in 2D meshes or grids.
    ///
    /// Example usage:
    /// 
    ///    mesh.reorder_by_morton_order(&mut elements);  
    ///
    pub fn reorder_by_morton_order(&mut self, elements: &mut [(u32, u32)]) {
        elements.par_sort_by_key(|&(x, y)| Self::morton_order_2d(x, y));
    }

    /// Computes the Morton order (Z-order curve) for a 2D point with coordinates (x, y).  
    ///
    /// This function interleaves the bits of the x and y coordinates to generate  
    /// a single value that represents the Morton order.  
    ///
    /// Example usage:
    /// 
    ///    let morton_order = Mesh::morton_order_2d(10, 20);  
    ///
    pub fn morton_order_2d(x: u32, y: u32) -> u64 {
        // Helper function to interleave the bits of a 32-bit integer.
        fn part1by1(n: u32) -> u64 {
            let mut n = n as u64;
            n = (n | (n << 16)) & 0x0000_0000_ffff_0000;
            n = (n | (n << 8)) & 0x0000_ff00_00ff_0000;
            n = (n | (n << 4)) & 0x00f0_00f0_00f0_00f0;
            n = (n | (n << 2)) & 0x0c30_0c30_0c30_0c30;
            n = (n | (n << 1)) & 0x2222_2222_2222_2222;
            n
        }

        // Interleave the bits of x and y to compute the Morton order.
        part1by1(x) | (part1by1(y) << 1)
    }
}

Analyze the contents of each of the attached files and commit them to memory. Describe and develop one integration test for the overall module. Below are the respective contents of each component of the `Mesh` module:

1. `src/domain/mesh/mod.rs`

```rust

pub mod entities;
pub mod geometry;
pub mod reordering;
pub mod boundary;
pub mod hierarchical;

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use crossbeam::channel::{Sender, Receiver};
use entities::*;
use geometry::*;
use reordering::*;
use boundary::*;
use hierarchical::*;

// Delegate methods to corresponding modules

#[derive(Clone)]
pub struct Mesh {
    pub sieve: Arc<Sieve>,
    pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,
    boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,
    boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,
}

impl Mesh {
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam::channel::unbounded();
        Mesh {
            sieve: Arc::new(Sieve::new()),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: Some(sender),
            boundary_data_receiver: Some(receiver),
        }
    }

    
}

#[cfg(test)]
pub mod tests;

```

2. `src/domain/mesh/entities.rs`

```rust

use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::RwLock;

impl Mesh {
    pub fn add_entity(&self, entity: MeshEntity) {
        self.entities.write().unwrap().insert(entity);
    }

    pub fn add_relationship(&mut self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    pub fn set_vertex_coordinates(&mut self, vertex_id: usize, coords: [f64; 3]) {
        self.vertex_coordinates.insert(vertex_id, coords);
        self.add_entity(MeshEntity::Vertex(vertex_id));
    }

    pub fn get_vertex_coordinates(&self, vertex_id: usize) -> Option<[f64; 3]> {
        self.vertex_coordinates.get(&vertex_id).cloned()
    }

    pub fn count_entities(&self, entity_type: &MeshEntity) -> usize {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| match (e, entity_type) {
                (MeshEntity::Vertex(_), MeshEntity::Vertex(_)) => true,
                (MeshEntity::Cell(_), MeshEntity::Cell(_)) => true,
                (MeshEntity::Edge(_), MeshEntity::Edge(_)) => true,
                (MeshEntity::Face(_), MeshEntity::Face(_)) => true,
                _ => false,
            })
            .count()
    }

    // Apply a function to all entities in parallel
    pub fn par_for_each_entity<F>(&self, func: F)
    where
        F: Fn(&MeshEntity) + Sync + Send,
    {
        let entities = self.entities.read().unwrap();
        entities.par_iter().for_each(func);
    }

    /// Get all cells in the mesh
    pub fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    /// Get all faces in the mesh
    pub fn get_faces(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .collect()
    }

    // Example method to compute some property for each entity in parallel
    pub fn compute_properties<F, PropertyType>(&self, compute_fn: F) -> FxHashMap<MeshEntity, PropertyType>
    where
        F: Fn(&MeshEntity) -> PropertyType + Sync + Send,
        PropertyType: Send,
    {
        let entities = self.entities.read().unwrap();
        entities
            .par_iter()
            .map(|entity| (*entity, compute_fn(entity)))
            .collect()
    }
}

```

3. `src/domain/mesh/geometry.rs`

```rust

use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use rustc_hash::FxHashSet;

impl Mesh {
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<FxHashSet<MeshEntity>> {
        self.sieve.cone(cell).map(|set| set.clone())
    }

    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> FxHashSet<MeshEntity> {
        self.sieve.support(face)
    }

    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Get face area (requires geometric data)
    pub fn get_face_area(&self, face: &MeshEntity) -> f64 {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
        };
        let geometry = Geometry::new();
        geometry.compute_face_area(face_shape, &face_vertices)
    }

    /// Get cell centroid
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
        let cell_vertices = self.get_cell_vertices(cell);
        let cell_shape = match cell_vertices.len() {
            4 => CellShape::Tetrahedron,
            5 => CellShape::Pyramid,
            6 => CellShape::Prism,
            8 => CellShape::Hexahedron,
            _ => panic!("Unsupported cell shape with {} vertices", cell_vertices.len()),
        };
        let geometry = Geometry::new();
        geometry.compute_cell_centroid(cell_shape, &cell_vertices)
    }

    /// Get cell vertices
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
        if let Some(connected_faces) = self.sieve.cone(cell) {
            for face in connected_faces {
                let face_vertices = self.get_face_vertices(&face);
                vertices.extend(face_vertices);
            }
            vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vertices.dedup();
        }
        vertices
    }

    /// Get face vertices
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertices.push(coords);
                    } else {
                        panic!("Coordinates for vertex {} not found", vertex_id);
                    }
                }
            }
        }
        vertices
    }

    pub fn get_neighboring_vertices(&self, vertex: &MeshEntity) -> Vec<MeshEntity> {
        let mut neighbors = FxHashSet::default();
        let connected_cells = self.sieve.support(vertex);

        for cell in &connected_cells {
            if let Some(cell_vertices) = self.sieve.cone(cell).as_ref() {
                for v in cell_vertices {
                    if v != vertex && matches!(v, MeshEntity::Vertex(_)) {
                        neighbors.insert(*v);
                    }
                }
            } else {
                panic!("Cell {:?} has no connected vertices", cell);
            }
        }
        neighbors.into_iter().collect()
    }

    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

}

```

4. `src/domain/mesh/reordering.rs`

```rust

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

```

5. `src/domain/mesh/boundary.rs`

```rust

use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crossbeam::channel::{Sender, Receiver};
use std::sync::RwLock;

impl Mesh {
    pub fn sync_boundary_data(&mut self) {
        self.send_boundary_data();
        self.receive_boundary_data();
    }

    pub fn set_boundary_channels(
        &mut self,
        sender: Sender<FxHashMap<MeshEntity, [f64; 3]>>,
        receiver: Receiver<FxHashMap<MeshEntity, [f64; 3]>>,
    ) {
        self.boundary_data_sender = Some(sender);
        self.boundary_data_receiver = Some(receiver);
    }

    pub fn receive_boundary_data(&mut self) {
        if let Some(ref receiver) = self.boundary_data_receiver {
            if let Ok(boundary_data) = receiver.recv() {
                let mut entities = self.entities.write().unwrap();
                for (entity, coords) in boundary_data {
                    if let MeshEntity::Vertex(id) = entity {
                        self.vertex_coordinates.insert(id, coords);
                    }
                    entities.insert(entity);
                }
            }
        }
    }

    pub fn send_boundary_data(&self) {
        if let Some(ref sender) = self.boundary_data_sender {
            let mut boundary_data = FxHashMap::default();
            let entities = self.entities.read().unwrap();
            for entity in entities.iter() {
                if let MeshEntity::Vertex(id) = entity {
                    if let Some(coords) = self.vertex_coordinates.get(id) {
                        boundary_data.insert(*entity, *coords);
                    }
                }
            }

            if let Err(e) = sender.send(boundary_data) {
                eprintln!("Failed to send boundary data: {:?}", e);
            }
        }
    }
}

```

6. `src/domain/mesh/hierarchical.rs`

```rust

use crate::domain::mesh_entity::MeshEntity;
use std::boxed::Box;

/// Represents a hierarchical mesh node, which can either be a leaf (non-refined)
/// or a branch (refined into smaller child elements).
#[derive(Debug, PartialEq)]
pub enum MeshNode<T> {
    Leaf(T),
    Branch {
        data: T,
        children: Box<[MeshNode<T>; 4]>, // 2D quadtree; change to `[MeshNode<T>; 8]` for 3D.
    },
}

impl<T: Clone> MeshNode<T> {
    /// Refines a leaf node into a branch with initialized child nodes.
    pub fn refine<F>(&mut self, init_child_data: F)
    where
        F: Fn(&T) -> [T; 4],  // Function to generate child data from the parent.
    {
        if let MeshNode::Leaf(data) = self {
            let children = init_child_data(data);
            *self = MeshNode::Branch {
                data: data.clone(),
                children: Box::new([
                    MeshNode::Leaf(children[0].clone()),
                    MeshNode::Leaf(children[1].clone()),
                    MeshNode::Leaf(children[2].clone()),
                    MeshNode::Leaf(children[3].clone()),
                ]),
            };
        }
    }

    /// Coarsens a branch back into a leaf node by collapsing children.
    pub fn coarsen(&mut self) {
        if let MeshNode::Branch { data, .. } = self {
            *self = MeshNode::Leaf(data.clone());
        }
    }

    /// Apply constraints at hanging nodes to ensure continuity between parent and child elements.
    pub fn apply_hanging_node_constraints(&self, parent_dofs: &mut [f64], child_dofs: &mut [[f64; 4]; 4]) {
        if let MeshNode::Branch { .. } = self {
            for i in 0..parent_dofs.len() {
                parent_dofs[i] = child_dofs.iter().map(|d| d[i]).sum::<f64>() / 4.0;
            }
        }
    }

    /// An iterator over all leaf nodes in the mesh hierarchy.
    pub fn leaf_iter(&self) -> LeafIterator<T> {
        LeafIterator { stack: vec![self] }
    }
}

/// Iterator for traversing through leaf nodes in the hierarchical mesh.
pub struct LeafIterator<'a, T> {
    stack: Vec<&'a MeshNode<T>>,
}

impl<'a, T> Iterator for LeafIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            match node {
                MeshNode::Leaf(data) => return Some(data),
                MeshNode::Branch { children, .. } => {
                    self.stack.extend(&**children);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::hierarchical::MeshNode;

    #[test]
    fn test_refine_leaf() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);

        if let MeshNode::Branch { children, .. } = node {
            assert_eq!(children[0], MeshNode::Leaf(11));
            assert_eq!(children[1], MeshNode::Leaf(12));
            assert_eq!(children[2], MeshNode::Leaf(13));
            assert_eq!(children[3], MeshNode::Leaf(14));
        } else {
            panic!("Node should have been refined to a branch.");
        }
    }

    #[test]
    fn test_coarsen_branch() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);
        node.coarsen();

        assert_eq!(node, MeshNode::Leaf(10));
    }

    #[test]
    fn test_apply_hanging_node_constraints() {
        let node = MeshNode::Branch {
            data: 0,
            children: Box::new([
                MeshNode::Leaf(1),
                MeshNode::Leaf(2),
                MeshNode::Leaf(3),
                MeshNode::Leaf(4),
            ]),
        };

        let mut parent_dofs = [0.0; 4];
        let mut child_dofs = [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ];

        node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);

        assert_eq!(parent_dofs, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_leaf_iterator() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);

        let leaves: Vec<_> = node.leaf_iter().collect();
        assert_eq!(leaves, [&11, &12, &13, &14]);
    }
}

```

7. `src/domain/mesh/tests.rs`

```rust

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;
    use crossbeam::channel::unbounded;
    use crate::domain::mesh::Mesh;

    #[test]
    fn test_send_receive_boundary_data() {
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        // Set up vertex coordinates.
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        mesh.vertex_coordinates.insert(2, [4.0, 5.0, 6.0]);

        // Add boundary entities.
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);

        // Set up a separate sender and receiver for testing.
        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        // Simulate sending the boundary data.
        mesh.send_boundary_data();

        // Create a second mesh instance to simulate the receiver.
        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(mesh.boundary_data_sender.clone().unwrap(), mesh.boundary_data_receiver.clone().unwrap());

        // Simulate receiving the boundary data.
        mesh_receiver.receive_boundary_data();

        // Verify that the receiver mesh has the updated vertex coordinates.
        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[1.0, 2.0, 3.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[4.0, 5.0, 6.0]));
    }

    /* #[test]
    fn test_receive_empty_data() {
        let mut mesh = Mesh::new();
        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        // Simulate receiving without sending any data.
        mesh.receive_boundary_data();

        // Ensure no data has been added.
        assert!(mesh.vertex_coordinates.is_empty());
    } */

    #[test]
    fn test_send_without_receiver() {
        let mut mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(3);
        mesh.vertex_coordinates.insert(3, [7.0, 8.0, 9.0]);
        mesh.add_entity(vertex);

        // Simulate sending the boundary data without setting a receiver.
        mesh.send_boundary_data();

        // No receiver to process, but this should not panic or fail.
        assert!(mesh.vertex_coordinates.get(&3).is_some());
    }

    #[test]
    fn test_add_entity() {
        let mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(1);
        mesh.add_entity(vertex);
        assert!(mesh.entities.read().unwrap().contains(&vertex));
    }

    #[test]
    fn test_iter_vertices() {
        let mut mesh = Mesh::new();
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        let vertices: Vec<_> = mesh.iter_vertices().collect();
        assert_eq!(vertices, vec![&1]);
    }
}
```
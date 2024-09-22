// src/domain/dm_mesh.rs

use crate::domain::{DPoint, DPointType};
use crate::domain::Section;
/// Represents the DM Mesh
// The Mesh structure: stores all dpoints and their relations.
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug)]
pub struct Mesh {
    pub dpoints: Vec<DPoint>,                          // A collection of DPoints
    pub index_ranges: HashMap<DPointType, (usize, usize)>, // Index ranges for each DPoint type
    pub sections: HashMap<DPointType, Section>,       // Section layout for each DPoint type
}

impl Mesh {
    // Create a new, empty Mesh
    pub fn new() -> Self {
        Mesh {
            dpoints: Vec::new(),
            index_ranges: HashMap::new(),
            sections: HashMap::new(),
        }
    }

    // Add a DPoint to the mesh and update index ranges for its type
    pub fn add_point(&mut self, id: usize, dpoint_type: DPointType) {
        self.dpoints.push(DPoint::new(id, dpoint_type.clone(), 1));

        // Update index ranges for the dpoint type
        if let Some((start, end)) = self.index_ranges.get_mut(&dpoint_type) {
            *end += 1;  // Expand the end range to include the new DPoint
        } else {
            self.index_ranges.insert(dpoint_type.clone(), (id, id + 1));
        }

        // Ensure a Section exists for this DPoint type
        self.sections.entry(dpoint_type.clone()).or_insert_with(|| Section::new(1));
    }

    // Retrieve a reference to a DPoint by its ID
    pub fn get_point(&self, id: usize) -> Option<&DPoint> {
        self.dpoints.get(id)
    }

    // Retrieve a mutable reference to a DPoint by its ID
    pub fn get_point_mut(&mut self, id: usize) -> Option<&mut DPoint> {
        self.dpoints.get_mut(id)
    }

    // Get the index range for a specific DPointType
    pub fn get_index_range(&self, dpoint_type: DPointType) -> Option<(usize, usize)> {
        self.index_ranges.get(&dpoint_type).copied()
    }

    // Initialize Section for a given DPoint type
    pub fn initialize_section(&mut self, dpoint_type: DPointType, num_fields: usize) {
        let section = Section::new(num_fields);
        self.sections.insert(dpoint_type, section);
    }

    // Set values in the section for a given DPoint and its type
    pub fn set_values_for_point(&mut self, point_id: usize, dpoint_type: DPointType, values: &[f64]) {
        if let Some(section) = self.sections.get_mut(&dpoint_type) {
            section.set_section(point_id, values.len());
        } else {
            panic!("Section for DPoint type {:?} not found", dpoint_type);
        }
    }

    // Get values from the section for a given DPoint and its type
    pub fn get_values_for_point(&self, point_id: usize, dpoint_type: DPointType) -> Option<Vec<f64>> {
        if let Some(section) = self.sections.get(&dpoint_type) {
            section.get_offset(point_id).map(|offset| {
                // Assuming the Field has a mechanism to fetch the values from the global array
                // Placeholder logic for demonstration
                vec![0.0; section.get_size(point_id).unwrap()]
            })
        } else {
            None
        }
    }

    // Reorder the DPoints in the mesh to follow the Hasse diagram based on dependencies
    pub fn reorder_points(&mut self) {
        // Set to track visited DPoints
        let mut visited = HashSet::new();

        // Queue to handle the reordering process (for BFS/DFS)
        let mut reordered_points = Vec::new();
        let mut point_queue = VecDeque::new();

        // Start with DPoints that have no dependencies (empty cone)
        for dpoint in &self.dpoints {
            if dpoint.cone.is_empty() {
                point_queue.push_back(dpoint.id);
            }
        }

        // Perform a topological sort (Hasse diagram traversal)
        while let Some(point_id) = point_queue.pop_front() {
            if !visited.contains(&point_id) {
                // Mark the DPoint as visited
                visited.insert(point_id);

                // Add the DPoint to the reordered list
                reordered_points.push(self.dpoints[point_id].clone());

                // For each DPoint that this DPoint covers (in its support)
                for covered_point in &self.dpoints[point_id].support {
                    // Add DPoints that depend on this DPoint to the queue if all their dependencies are satisfied
                    if self.dpoints[*covered_point]
                        .cone
                        .iter()
                        .all(|&dep_id| visited.contains(&dep_id))
                    {
                        point_queue.push_back(*covered_point);
                    }
                }
            }
        }

        // Update the mesh with the reordered DPoints
        self.dpoints = reordered_points;
    }
}

// Test module for Mesh structure
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{DPoint, DPointType};

    #[test]
    fn test_add_and_get_point() {
        let mut mesh = Mesh::new();
        
        // Add a vertex with ID 0
        mesh.add_point(0, DPointType::Vertex);
        
        // Add an edge with ID 1
        mesh.add_point(1, DPointType::Edge);
        
        // Retrieve and verify the vertex
        if let Some(dpoint) = mesh.get_point(0) {
            assert_eq!(dpoint.id, 0);
            assert_eq!(dpoint.point_type, DPointType::Vertex);
        } else {
            panic!("DPoint with ID 0 not found.");
        }

        // Retrieve and verify the edge
        if let Some(dpoint) = mesh.get_point(1) {
            assert_eq!(dpoint.id, 1);
            assert_eq!(dpoint.point_type, DPointType::Edge);
        } else {
            panic!("DPoint with ID 1 not found.");
        }
    }

    #[test]
    fn test_index_ranges() {
        let mut mesh = Mesh::new();
        
        // Add DPoints to the mesh
        mesh.add_point(0, DPointType::Vertex);
        mesh.add_point(1, DPointType::Edge);
        mesh.add_point(2, DPointType::Vertex);
        
        // Check index ranges for vertices and edges
        assert_eq!(mesh.get_index_range(DPointType::Vertex), Some((0, 3)));
        assert_eq!(mesh.get_index_range(DPointType::Edge), Some((1, 2)));
    }

    #[test]
    fn test_reorder_points() {
        let mut mesh = Mesh::new();

        // Add DPoints to the mesh with dependencies (cone and support relationships)
        // Vertex (id 0) is covered by Edge (id 1)
        let mut vertex = DPoint::new(0, DPointType::Vertex, 1);
        let mut edge = DPoint::new(1, DPointType::Edge, 1);

        // Edge 1 is covered by Vertex 0
        edge.set_cone(vec![0]);
        vertex.set_support(vec![1]);

        mesh.dpoints.push(vertex);
        mesh.dpoints.push(edge);

        // Initially, DPoints are added in reverse order (Edge before Vertex)
        assert_eq!(mesh.dpoints[0].id, 0);  // Vertex
        assert_eq!(mesh.dpoints[1].id, 1);  // Edge

        // Reorder DPoints to follow the Hasse diagram (Vertex should come before Edge)
        mesh.reorder_points();

        // Verify that the DPoints have been reordered correctly
        assert_eq!(mesh.dpoints[0].id, 0);  // Vertex
        assert_eq!(mesh.dpoints[1].id, 1);  // Edge
    }
}

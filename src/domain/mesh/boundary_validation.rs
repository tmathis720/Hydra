use rustc_hash::FxHashMap;
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::mesh::Mesh;

use super::entities;

/// The `BoundaryValidation` struct encapsulates validation routines for boundary conditions 
/// on a mesh, ensuring correct application and consistency across partitions.
pub struct BoundaryValidation<'a> {
    mesh: &'a Mesh,
}

impl<'a> BoundaryValidation<'a> {
    /// Creates a new `BoundaryValidation` instance associated with a mesh.
    pub fn new(mesh: &'a Mesh) -> Self {
        BoundaryValidation { mesh }
    }

    /// Validates that boundary conditions are only applied to entities explicitly tagged as boundaries.
    pub fn validate_boundary_condition_application(&self) -> Result<(), String> {
        let entities = self.mesh.entities.read().unwrap();
        for entity in entities.iter() {
            if let MeshEntity::Face(id) = entity {
                // Example check: Verify each face tagged for boundary condition has the necessary data
                if !self.mesh.vertex_coordinates.contains_key(id) {
                    return Err(format!("Boundary condition missing for face entity with ID {}", id));
                }
            }
        }
        Ok(())
    }

    /// Synchronizes boundary data across partitions, ensuring consistency.
    /// 
    /// This method assumes that data has been sent and received through the mesh’s boundary channels.
    pub fn validate_boundary_data_synchronization(&self) -> Result<(), String> {
        if let Some(ref receiver) = self.mesh.boundary_data_receiver {
            match receiver.try_recv() {
                Ok(boundary_data) => {
                    let entities = self.mesh.entities.read().unwrap();
                    for (entity, coords) in boundary_data.iter() {
                        if let MeshEntity::Vertex(id) = entity {
                            if let Some(local_coords) = self.mesh.vertex_coordinates.get(id) {
                                if *local_coords != *coords {
                                    return Err(format!("Mismatched boundary data for vertex ID {}: local {:?} vs received {:?}", id, local_coords, coords));
                                }
                            } else {
                                return Err(format!("Vertex ID {} missing in local mesh for synchronization", id));
                            }
                        }
                    }
                    Ok(())
                }
                Err(_) => Err("Failed to receive boundary data for synchronization".to_string()),
            }
        } else {
            Err("Boundary data receiver channel is not set".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::Mesh;
    use crate::domain::mesh_entity::MeshEntity;
    use crossbeam::channel;

    #[test]
    fn test_boundary_condition_application() {
        let mut mesh = Mesh::new();
        let boundary_validation = BoundaryValidation::new(&mesh);
        
        // Simulate setting up a boundary condition on a face
        mesh.entities.write().unwrap().insert(MeshEntity::Face(1));
        mesh.vertex_coordinates.insert(1, [0.0, 0.0, 0.0]);

        // Validate boundary condition application
        assert!(boundary_validation.validate_boundary_condition_application().is_ok());
    }

    #[test]
    fn test_boundary_data_synchronization() {
        let mut mesh = Mesh::new();
        let boundary_validation = BoundaryValidation::new(&mesh);

        // Setup boundary communication channels
        let (sender, receiver) = channel::unbounded();
        mesh.set_boundary_channels(sender, receiver.clone());

        // Simulate boundary data transmission
        let mut boundary_data = FxHashMap::default();
        boundary_data.insert(MeshEntity::Vertex(1), [1.0, 2.0, 3.0]);
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        receiver.send(boundary_data).unwrap();

        // Validate boundary data synchronization
        assert!(boundary_validation.validate_boundary_data_synchronization().is_ok());
    }
}

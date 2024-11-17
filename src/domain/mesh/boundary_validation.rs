use crate::domain::mesh_entity::MeshEntity;
use crate::domain::mesh::Mesh;
use std::time::Duration;

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
                if !self.mesh.vertex_coordinates.contains_key(id) {
                    return Err(format!(
                        "Boundary condition missing for face entity with ID {}",
                        id
                    ));
                }
            }
        }
        Ok(())
    }

    /// Synchronizes boundary data across partitions, ensuring consistency.
    pub fn validate_boundary_data_synchronization(&self) -> Result<(), String> {
        if self.mesh.boundary_data_receiver.is_none() {
            return Err("Boundary data receiver channel is not set".to_string());
        }

        if let Some(ref receiver) = self.mesh.boundary_data_receiver {
            match receiver.recv_timeout(Duration::from_millis(200)) {
                Ok(boundary_data) => {
                    let _entities = self.mesh.entities.read().unwrap();
                    for (entity, coords) in boundary_data.iter() {
                        if let MeshEntity::Vertex(id) = entity {
                            if let Some(local_coords) = self.mesh.vertex_coordinates.get(id) {
                                if *local_coords != *coords {
                                    return Err(format!(
                                        "Mismatched boundary data for vertex ID {}: local {:?} vs received {:?}",
                                        id, local_coords, coords
                                    ));
                                }
                            } else {
                                return Err(format!(
                                    "Vertex ID {} missing in local mesh for synchronization",
                                    id
                                ));
                            }
                        }
                    }
                    Ok(())
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    Err("Timeout while waiting for boundary data".to_string())
                }
                Err(e) => Err(format!("Failed to receive boundary data: {:?}", e)),
            }
        } else {
            Err("Boundary data receiver channel is not set".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{domain::mesh::Mesh, Sieve};
    use crate::domain::mesh_entity::MeshEntity;
    use rustc_hash::{FxHashMap, FxHashSet};
    use crossbeam::channel::{unbounded, Sender, Receiver};
    use std::sync::{Arc, RwLock};

    /// Helper function to create a mock mesh.
    fn create_mock_mesh() -> (Mesh, Sender<FxHashMap<MeshEntity, [f64; 3]>>, Receiver<FxHashMap<MeshEntity, [f64; 3]>>) {
        let (sender, receiver) = unbounded();
        let mesh = Mesh {
            sieve: Sieve::new().into(),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: Some(sender.clone()),
            boundary_data_receiver: Some(receiver.clone()),
        };
        (mesh, sender, receiver)
    }

    #[test]
    fn test_validate_boundary_condition_application() {
        let (mut mesh, _sender, _receiver) = create_mock_mesh();

        // Add a face entity without boundary data
        mesh.entities.write().unwrap().insert(MeshEntity::Face(1));

        // First validation attempt
        {
            let validator = BoundaryValidation::new(&mesh);
            let result = validator.validate_boundary_condition_application();
            assert!(result.is_err());
            assert_eq!(
                result.unwrap_err(),
                "Boundary condition missing for face entity with ID 1"
            );
        } // End the immutable borrow of `mesh` here.

        // Add boundary data and re-validate
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);

        // Second validation attempt
        {
            let validator = BoundaryValidation::new(&mesh);
            let result = validator.validate_boundary_condition_application();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_validate_boundary_data_synchronization() {
        let (mut mesh, sender, _receiver) = create_mock_mesh();

        // Add a vertex entity with local coordinates
        mesh.entities.write().unwrap().insert(MeshEntity::Vertex(1));
        mesh.vertex_coordinates.insert(1, [0.0, 1.0, 2.0]);

        // Simulate synchronized data
        let mut sync_data = FxHashMap::default();
        sync_data.insert(MeshEntity::Vertex(1), [0.0, 1.0, 2.0]);
        sender.send(sync_data).unwrap();

        let validator = BoundaryValidation::new(&mesh);

        // Validation should pass since data is consistent
        let result = validator.validate_boundary_data_synchronization();
        assert!(result.is_ok());

        // Simulate mismatched data
        let mut inconsistent_data = FxHashMap::default();
        inconsistent_data.insert(MeshEntity::Vertex(1), [1.0, 2.0, 3.0]);
        sender.send(inconsistent_data).unwrap();

        // Validation should fail
        let result = validator.validate_boundary_data_synchronization();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Mismatched boundary data for vertex ID 1: local [0.0, 1.0, 2.0] vs received [1.0, 2.0, 3.0]"
        );
    }
}

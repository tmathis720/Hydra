use crate::domain::mesh_entity::MeshEntity;
use crate::domain::mesh::Mesh;
use std::time::Duration;

/// `BoundaryValidation` encapsulates routines for validating boundary conditions and data synchronization.
///
/// This struct provides functionality to ensure:
/// 1. Boundary conditions are applied only to properly tagged mesh entities.
/// 2. Boundary data is consistent across mesh partitions.
///
/// These validations are critical for ensuring the integrity of distributed mesh simulations.
pub struct BoundaryValidation<'a> {
    /// A reference to the mesh being validated.
    mesh: &'a Mesh,
}

impl<'a> BoundaryValidation<'a> {
    /// Constructs a new `BoundaryValidation` instance associated with a specific mesh.
    ///
    /// # Arguments
    /// - `mesh`: A reference to the mesh on which validation operations will be performed.
    pub fn new(mesh: &'a Mesh) -> Self {
        BoundaryValidation { mesh }
    }

    /// Validates that boundary conditions are applied only to properly tagged boundary entities.
    ///
    /// This method iterates over all face entities in the mesh and checks if each face
    /// has corresponding boundary data (e.g., vertex coordinates). If any face entity lacks
    /// this data, the method returns an error identifying the problematic face.
    ///
    /// # Returns
    /// - `Ok(())` if all boundary conditions are correctly applied.
    /// - `Err(String)` if any face entity is missing the required boundary data.
    pub fn validate_boundary_condition_application(&self) -> Result<(), String> {
        let entities = self.mesh.entities.read().unwrap();

        // Iterate over all mesh entities
        for entity in entities.iter() {
            // Check only face entities
            if let MeshEntity::Face(id) = entity {
                // Validate that the face has boundary data
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

    /// Validates the consistency of boundary data synchronization across partitions.
    ///
    /// This method compares locally stored boundary data (e.g., vertex coordinates)
    /// with data received via the boundary data receiver channel. If discrepancies
    /// are detected or if the receiver is not configured, an appropriate error is returned.
    ///
    /// # Returns
    /// - `Ok(())` if the boundary data is consistent across partitions.
    /// - `Err(String)` if inconsistencies are found or if communication fails.
    pub fn validate_boundary_data_synchronization(&self) -> Result<(), String> {
        // Ensure the receiver channel is configured
        if self.mesh.boundary_data_receiver.is_none() {
            return Err("Boundary data receiver channel is not set".to_string());
        }

        if let Some(ref receiver) = self.mesh.boundary_data_receiver {
            // Attempt to receive boundary data with a timeout
            match receiver.recv_timeout(Duration::from_millis(200)) {
                Ok(boundary_data) => {
                    // Access the local entities set for validation
                    let _entities = self.mesh.entities.read().unwrap();

                    // Validate consistency for each received vertex
                    for (entity, coords) in boundary_data.iter() {
                        if let MeshEntity::Vertex(id) = entity {
                            // Check if the local mesh has the vertex
                            if let Some(local_coords) = self.mesh.vertex_coordinates.get(id) {
                                // Compare local and received coordinates
                                if *local_coords != *coords {
                                    return Err(format!(
                                        "Mismatched boundary data for vertex ID {}: local {:?} vs received {:?}",
                                        id, local_coords, coords
                                    ));
                                }
                            } else {
                                // Report missing local vertex
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
                    // Report a timeout error
                    Err("Timeout while waiting for boundary data".to_string())
                }
                Err(e) => {
                    // Handle other reception errors
                    Err(format!("Failed to receive boundary data: {:?}", e))
                }
            }
        } else {
            // Report an unconfigured receiver channel
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

    /// Creates a mock mesh with boundary data communication channels for testing purposes.
    ///
    /// # Returns
    /// - A tuple containing the mock `Mesh`, a `Sender` for transmitting boundary data,
    ///   and a `Receiver` for receiving boundary data.
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

        // Perform initial validation, expecting an error
        {
            let validator = BoundaryValidation::new(&mesh);
            let result = validator.validate_boundary_condition_application();
            assert!(result.is_err());
            assert_eq!(
                result.unwrap_err(),
                "Boundary condition missing for face entity with ID 1"
            );
        }

        // Add boundary data for the face entity
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);

        // Re-validate, expecting success
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

        // Validate synchronized data, expecting success
        let result = validator.validate_boundary_data_synchronization();
        assert!(result.is_ok());

        // Simulate mismatched data
        let mut inconsistent_data = FxHashMap::default();
        inconsistent_data.insert(MeshEntity::Vertex(1), [1.0, 2.0, 3.0]);
        sender.send(inconsistent_data).unwrap();

        // Validate mismatched data, expecting an error
        let result = validator.validate_boundary_data_synchronization();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Mismatched boundary data for vertex ID 1: local [0.0, 1.0, 2.0] vs received [1.0, 2.0, 3.0]"
        );
    }
}

use crate::domain::mesh_entity::MeshEntity;
use std::collections::HashMap;
use nalgebra::DVector;

pub struct NeumannBC {
    pub fluxes: HashMap<MeshEntity, f64>,  // Map boundary entities to their flux values
}

impl NeumannBC {
    /// Creates a new NeumannBC structure
    pub fn new() -> Self {
        NeumannBC {
            fluxes: HashMap::new(),
        }
    }

    /// Set a Neumann boundary condition (flux) for a given mesh entity
    pub fn set_bc(&mut self, entity: MeshEntity, flux: f64) {
        self.fluxes.insert(entity, flux);
    }

    /// Check if a mesh entity has a Neumann boundary condition
    pub fn is_bc(&self, entity: &MeshEntity) -> bool {
        self.fluxes.contains_key(entity)
    }

    /// Get the Neumann flux value for a mesh entity
    pub fn get_flux(&self, entity: &MeshEntity) -> f64 {
        *self.fluxes.get(entity).expect(&format!(
            "Neumann BC not set for entity {:?}",
            entity
        ))
    }

    /// Apply the Neumann boundary condition to the RHS vector
    ///
    /// `rhs`: The right-hand side vector (DVector)
    /// `entity_to_index`: Mapping from MeshEntity to vector index
    /// `face_areas`: Map from face IDs to face areas
    pub fn apply_bc(
        &self,
        rhs: &mut DVector<f64>,
        face_to_cell_index: &HashMap<MeshEntity, usize>,
        face_areas: &HashMap<usize, f64>,
    ) {
        for (face_entity, &flux) in &self.fluxes {
            if let Some(&cell_index) = face_to_cell_index.get(face_entity) {
                if let MeshEntity::Face(face_id) = face_entity {
                    let area = face_areas.get(face_id).expect(&format!(
                        "Area not found for face {}",
                        face_id
                    ));
                    rhs[cell_index] += flux * area;
                } else {
                    panic!("Neumann BC should be applied to Face entities, got {:?}", face_entity);
                }
            } else {
                panic!("Face entity {:?} not found in face_to_cell_index mapping", face_entity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;
    use nalgebra::DVector;
    use std::collections::HashMap;

    #[test]
    fn test_neumann_bc() {
        // Create a mock RHS vector
        let mut rhs = DVector::<f64>::zeros(5);

        // Create the NeumannBC structure
        let mut neumann_bc = NeumannBC::new();

        // Define a couple of boundary faces
        let boundary_face_1 = MeshEntity::Face(1);
        let boundary_face_2 = MeshEntity::Face(3);

        // Set Neumann boundary conditions (fluxes) for those faces
        neumann_bc.set_bc(boundary_face_1, 10.0);
        neumann_bc.set_bc(boundary_face_2, -5.0);

        // Create face to cell index mapping
        let mut face_to_cell_index = HashMap::new();
        face_to_cell_index.insert(boundary_face_1, 1); // Assuming Face(1) is adjacent to cell index 1
        face_to_cell_index.insert(boundary_face_2, 3); // Assuming Face(3) is adjacent to cell index 3

        // Create face areas mapping
        let mut face_areas = HashMap::new();
        face_areas.insert(1, 2.0); // Area of face 1
        face_areas.insert(3, 1.5); // Area of face 3

        // Apply the boundary conditions to the RHS vector
        neumann_bc.apply_bc(&mut rhs, &face_to_cell_index, &face_areas);

        // Check that the RHS vector was modified correctly
        assert_eq!(rhs[1], 10.0 * 2.0);  // 10.0 flux multiplied by area 2.0
        assert_eq!(rhs[3], -5.0 * 1.5);  // -5.0 flux multiplied by area 1.5
    }
}

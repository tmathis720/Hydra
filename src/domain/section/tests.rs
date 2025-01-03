
#[cfg(test)]
mod tests {
    use crate::{domain::{mesh_entity::MeshEntity, section::{Scalar, Tensor3x3, Vector3}}, Section, Vector};

    /// Helper function to create a `MeshEntity` for testing purposes.
    ///
    /// This function creates a `Vertex` variant of `MeshEntity` with the given ID.
    /// Adjust this function if other `MeshEntity` variants need to be tested.
    fn create_test_mesh_entity(id: usize) -> MeshEntity {
        MeshEntity::Vertex(id)
    }

    #[test]
    /// Tests the addition of two `Vector3` instances.
    ///
    /// Validates that component-wise addition is performed correctly.
    fn test_vector3_add() {
        let v1 = Vector3([1.0, 2.0, 3.0]);
        let v2 = Vector3([4.0, 5.0, 6.0]);
        let result = v1 + v2;

        assert_eq!(result.0, [5.0, 7.0, 9.0]);
    }

    #[test]
    /// Tests the `+=` operation for `Vector3`.
    ///
    /// Validates that component-wise addition is correctly applied in-place.
    fn test_vector3_add_assign() {
        let mut v1 = Vector3([1.0, 2.0, 3.0]);
        let v2 = Vector3([0.5, 0.5, 0.5]);
        v1 += v2;

        assert_eq!(v1.0, [1.5, 2.5, 3.5]);
    }

    #[test]
    /// Tests scalar multiplication for `Vector3`.
    ///
    /// Validates that each component of the vector is scaled correctly.
    fn test_vector3_mul() {
        let v = Vector3([1.0, 2.0, 3.0]);
        let scaled = v * 2.0;

        assert_eq!(scaled.0, [2.0, 4.0, 6.0]);
    }

    #[test]
    /// Tests the `+=` operation for `Tensor3x3`.
    ///
    /// Validates that component-wise addition is applied correctly to tensors.
    fn test_tensor3x3_add_assign() {
        let mut t1 = Tensor3x3([[1.0; 3]; 3]);
        let t2 = Tensor3x3([[0.5; 3]; 3]);
        t1 += t2;

        assert_eq!(t1.0, [[1.5; 3]; 3]);
    }

    #[test]
    /// Tests scalar multiplication for `Tensor3x3`.
    ///
    /// Validates that all components of the tensor are scaled correctly.
    fn test_tensor3x3_mul() {
        let t = Tensor3x3([[1.0; 3]; 3]);
        let scaled = t * 2.0;

        assert_eq!(scaled.0, [[2.0; 3]; 3]);
    }

    #[test]
    /// Tests setting and retrieving data in a `Section<Scalar>`.
    ///
    /// Ensures that values can be stored and accessed correctly by their associated `MeshEntity`.
    fn test_section_set_and_restrict_data() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        let value = Scalar(3.14);

        section.set_data(entity, value);
        let retrieved = section.restrict(&entity);

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, 3.14);
    }

    #[test]
    /// Tests parallel updates for a `Section<Scalar>`.
    ///
    /// Validates that all values in the section are updated correctly and efficiently in parallel.
    fn test_section_parallel_update() {
        let section: Section<Scalar> = Section::new();
        let entities: Vec<MeshEntity> = (1..=10).map(create_test_mesh_entity).collect();

        for (i, entity) in entities.iter().enumerate() {
            section.set_data(*entity, Scalar(i as f64));
        }

        section.parallel_update(|value| {
            value.0 *= 2.0; // Double each value
        });

        for (i, entity) in entities.iter().enumerate() {
            assert_eq!(section.restrict(entity).unwrap().0, (i as f64) * 2.0);
        }
    }

    #[test]
    /// Tests updating a `Section<Scalar>` with a derivative section.
    ///
    /// Ensures that the update correctly adds the scaled derivative to the section's values.
    fn test_section_update_with_derivative() {
        let section: Section<Scalar> = Section::new();
        let derivative: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);

        section.set_data(entity, Scalar(1.0));
        derivative.set_data(entity, Scalar(0.5));

        section.update_with_derivative(&derivative, 2.0); // Time step is 2.0

        assert_eq!(section.restrict(&entity).unwrap().0, 2.0);
    }

    #[test]
    /// Tests retrieving all `MeshEntity` objects from a `Section<Scalar>`.
    ///
    /// Ensures that all entities stored in the section are returned correctly.
    fn test_section_entities() {
        let section: Section<Scalar> = Section::new();
        let entities: Vec<MeshEntity> = (1..=5).map(create_test_mesh_entity).collect();

        for entity in &entities {
            section.set_data(*entity, Scalar(1.0));
        }

        let retrieved_entities = section.entities();
        assert_eq!(retrieved_entities.len(), entities.len());
    }

    #[test]
    /// Tests clearing all data from a `Section<Scalar>`.
    ///
    /// Validates that the section becomes empty after the clear operation.
    fn test_section_clear() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(1.0));

        section.clear();

        assert!(section.restrict(&entity).is_none());
    }

    #[test]
    /// Tests scaling all values in a `Section<Scalar>`.
    ///
    /// Ensures that all values are scaled correctly by the given factor.
    fn test_section_scale() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(2.0));

        section.scale(3.0); // Scale by 3.0

        assert_eq!(section.restrict(&entity).unwrap().0, 6.0);
    }

    /// Utility function for debugging `Section` contents during test failures.
    ///
    /// Prints all key-value pairs in the `Section` for inspection.
    /// This function is not part of the test suite but can be used for debugging purposes.
    fn debug_section_data<T>(section: &Section<T>)
    where
        T: std::fmt::Debug,
    {
        println!("Section data:");
        for entry in section.data.iter() {
            println!("{:?} -> {:?}", entry.key(), entry.value());
        }
    }

    #[test]
    /// Example test to demonstrate debugging output for `Section` contents.
    ///
    /// Useful for inspecting data during test failures.
    fn test_debugging_output() {
        let section: Section<Scalar> = Section::new();
        let entity = create_test_mesh_entity(1);
        section.set_data(entity, Scalar(1.0));

        debug_section_data(&section);
    }

    #[test]
    fn test_vector_trait_for_section() {
        use crate::domain::mesh_entity::MeshEntity;

        // Create a new section
        let mut section = Section::new();
        section.set_data(MeshEntity::Cell(0), Scalar(1.0));
        section.set_data(MeshEntity::Cell(1), Scalar(2.0));
        section.set_data(MeshEntity::Cell(2), Scalar(3.0));

        // Test `len`
        assert_eq!(section.len(), 3);

        // Test `get`
        assert_eq!(section.get(0), 1.0);
        assert_eq!(section.get(1), 2.0);
        assert_eq!(section.get(2), 3.0);

        // Test `set`
        section.set(1, 5.0);
        assert_eq!(section.get(1), 5.0);

        // Test `dot`
        let other = Section::new();
        other.set_data(MeshEntity::Cell(0), Scalar(2.0));
        other.set_data(MeshEntity::Cell(1), Scalar(3.0));
        other.set_data(MeshEntity::Cell(2), Scalar(4.0));
        assert_eq!(section.dot(&other), 1.0 * 2.0 + 5.0 * 3.0 + 3.0 * 4.0);
    }

}

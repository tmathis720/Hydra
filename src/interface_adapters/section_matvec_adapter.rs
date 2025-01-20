use crate::domain::section::{Scalar, Section};
use crate::domain::mesh_entity::MeshEntity;
use dashmap::DashMap;
use faer::mat::Mat;

/// Adapter for converting between `Section` and matrix/vector representations.
pub struct SectionMatVecAdapter;

impl SectionMatVecAdapter {
    /// Converts a `Section` to a dense vector (`Vec<f64>`) representation.
    ///
    /// # Parameters
    /// - `section`: A reference to the `Section<Scalar>` to convert.
    /// - `size`: The total number of entities (used for padding with zeros for missing indices).
    ///
    /// # Returns
    /// A dense vector (`Vec<f64>`) where each index corresponds to a `MeshEntity`.
    pub fn section_to_dense_vector(section: &Section<crate::domain::section::Scalar>, size: usize) -> Vec<f64> {
        let mut dense_vector = vec![0.0; size];

        for entry in section.data.iter() {
            let entity = entry.key();
            let scalar = entry.value();
            let index = entity.get_id(); // Use `get_id` to map `MeshEntity` to index
            dense_vector[index] = scalar.0;
        }

        dense_vector
    }

    /// Converts a dense vector (`Vec<f64>`) back into a `Section<Scalar>`.
    ///
    /// # Parameters
    /// - `vector`: A dense vector (`Vec<f64>`) to convert.
    /// - `entities`: A vector of `MeshEntity` objects corresponding to the indices of the vector.
    ///
    /// # Returns
    /// A `Section<Scalar>` where each entity is associated with a scalar value from the vector.
    pub fn dense_vector_to_section(
        vector: &[f64],
        entities: &[MeshEntity],
    ) -> Section<crate::domain::section::Scalar> {
        let section = Section::new();

        for (index, &value) in vector.iter().enumerate() {
            if let Some(entity) = entities.get(index) {
                section.set_data(*entity, crate::domain::section::Scalar(value));
            }
        }

        section
    }

    /// Converts a `Section` to a dense matrix (`Mat<f64>`) representation.
    ///
    /// # Parameters
    /// - `section`: A reference to the `Section<Tensor3x3>` to convert.
    /// - `size`: The total number of entities (used for padding with zeros for missing indices).
    ///
    /// # Returns
    /// A dense matrix (`Mat<f64>`) representing the section.
    pub fn section_to_dense_matrix(section: &Section<crate::domain::section::Tensor3x3>, size: usize) -> Mat<f64> {
        let mut matrix = Mat::zeros(size, size);

        for entry in section.data.iter() {
            let entity = entry.key();
            let tensor = entry.value();
            let index = entity.get_id(); // Use `get_id` for row/column indexing

            // Populate the diagonal with the tensor's components (simplified for demonstration)
            matrix.write(index, index, tensor.0[0][0]); // Assuming Tensor3x3 for diagonal
        }

        matrix
    }

    /// Converts a `Section` to a dense matrix (`MatMut<f64>`) representation.
    ///
    /// # Parameters
    /// - `section`: A reference to the `Section<Scalar>` to convert.
    /// - `size`: The total number of entities (used for padding with zeros for missing indices).
    ///
    /// # Returns
    /// A dense matrix (`Mat<f64>`) representing the section.
    pub fn section_to_matmut(
        section: &Section<Scalar>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        size: usize,
    ) -> Mat<f64> {
        // Validate that `size` is large enough to handle all indices in `entity_to_index`.
        for index in entity_to_index.iter().map(|entry| *entry.value()) {
            assert!(
                index < size,
                "Entity-to-index map contains an index ({}) larger than the specified size ({}).",
                index,
                size
            );
        }
    
        // Create a matrix with the correct dimensions
        let mut matrix = Mat::<f64>::zeros(size, 1);
    
        for entry in section.data.iter() {
            let entity = entry.key();
            let scalar = entry.value();
            if let Some(index) = entity_to_index.get(entity) {
                assert!(
                    *index < size,
                    "Index ({}) for entity {:?} is out of bounds (size: {}).",
                    *index,
                    entity,
                    size
                );
                matrix.write(*index, 0, scalar.0);
            } else {
                panic!("Entity {:?} not found in index map", entity);
            }
        }
    
        matrix
    }
    

    /// Converts a dense matrix (`Mat<f64>`) back into a `Section<Tensor3x3>`.
    ///
    /// # Parameters
    /// - `matrix`: A dense matrix (`Mat<f64>`) to convert.
    /// - `entities`: A vector of `MeshEntity` objects corresponding to the matrix rows/columns.
    ///
    /// # Returns
    /// A `Section<Tensor3x3>` representing the dense matrix.
    pub fn dense_matrix_to_section(
        matrix: &Mat<f64>,
        entities: &[MeshEntity],
    ) -> Section<crate::domain::section::Tensor3x3> {
        let section = Section::new();

        for (index, entity) in entities.iter().enumerate() {
            // Read diagonal element (simplified for demonstration)
            let value = matrix.read(index, index);
            let tensor = crate::domain::section::Tensor3x3([[value, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
            section.set_data(*entity, tensor);
        }

        section
    }

    /// Converts a mutable matrix view (`MatMut<f64>`) back into a `Section<Scalar>`.
    ///
    /// # Parameters
    /// - `mat`: A mutable matrix view (`MatMut<f64>`) to convert.
    /// - `entities`: A vector of `MeshEntity` corresponding to matrix rows/columns.
    ///
    /// # Returns
    /// - `Section<Scalar>` representing the matrix.
    pub fn matmut_to_section(
        mat: &Mat<f64>,
        entities: &[MeshEntity],
    ) -> Section<crate::domain::section::Scalar> {
        let section = Section::new();

        for (index, entity) in entities.iter().enumerate() {
            let value = mat.read(index, index); // Diagonal value
            section.set_data(*entity, crate::domain::section::Scalar(value));
        }

        section
    }

    /// Updates a `Section<Scalar>` using values from a dense vector (`Vec<f64>`).
    ///
    /// # Parameters
    /// - `section`: Mutable reference to the `Section<Scalar>` to update.
    /// - `vector`: Dense vector containing the new values.
    /// - `entities`: Mesh entities corresponding to the indices in the vector.
    pub fn update_section_with_vector(
        section: &mut Section<crate::domain::section::Scalar>,
        vector: &[f64],
        entities: &[MeshEntity],
    ) {
        for (index, &value) in vector.iter().enumerate() {
            if let Some(entity) = entities.get(index) {
                section.set_data(*entity, crate::domain::section::Scalar(value));
            }
        }
    }


    /// Converts a sparse matrix represented by a `Section<Scalar>` into a dense `Mat<f64>`.
    ///
    /// # Parameters
    /// - `section`: The sparse section to convert.
    /// - `size`: Size of the dense matrix (rows and columns).
    ///
    /// # Returns
    /// - `Mat<f64>` representing the dense version of the sparse matrix.
    pub fn sparse_to_dense_matrix(
        section: &Section<crate::domain::section::Scalar>,
        size: usize,
    ) -> Mat<f64> {
        let mut dense_mat = Mat::<f64>::zeros(size, size);

        for entry in section.data.iter() {
            let entity = entry.key();
            let scalar = entry.value();
            let idx = entity.get_id();
            dense_mat.write(idx, idx, scalar.0);
        }

        dense_mat
    }

    /// Updates a dense matrix (`Mat<f64>`) using data from a `Section<Scalar>`.
    ///
    /// # Parameters
    /// - `mat`: Mutable reference to the dense matrix.
    /// - `section`: Sparse section containing update values.
    pub fn update_dense_matrix_from_section(
        mat: &mut Mat<f64>,
        section: &Section<crate::domain::section::Scalar>,
    ) {
        for entry in section.data.iter() {
            let entity = entry.key();
            let scalar = entry.value();
            let idx = entity.get_id();
            mat.write(idx, idx, scalar.0);
        }
    }


}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::section::{Scalar, Section};
    use crate::domain::mesh_entity::MeshEntity;
    use dashmap::DashMap;
    use faer::mat::Mat;

    /// Helper function to create a `Section<Scalar>` for testing.
    fn create_test_section_scalar() -> Section<Scalar> {
        let section = Section::new();
        section.set_data(MeshEntity::Face(0), Scalar(1.0));
        section.set_data(MeshEntity::Face(1), Scalar(2.0));
        section.set_data(MeshEntity::Face(2), Scalar(3.0));
        section
    }

    /// Helper function to create an entity-to-index map.
    fn create_entity_to_index_map() -> DashMap<MeshEntity, usize> {
        let map = DashMap::new();
        map.insert(MeshEntity::Face(0), 0);
        map.insert(MeshEntity::Face(1), 1);
        map.insert(MeshEntity::Face(2), 2);
        map
    }

    /// Helper function to create a list of `MeshEntity` for testing.
    fn create_test_entities() -> Vec<MeshEntity> {
        vec![MeshEntity::Face(0), MeshEntity::Face(1), MeshEntity::Face(2)]
    }

    #[test]
    fn test_section_to_dense_vector() {
        let section = create_test_section_scalar();
        let dense_vector = SectionMatVecAdapter::section_to_dense_vector(&section, 3);

        assert_eq!(dense_vector.len(), 3);
        assert_eq!(dense_vector[0], 1.0);
        assert_eq!(dense_vector[1], 2.0);
        assert_eq!(dense_vector[2], 3.0);
    }

    #[test]
    fn test_dense_vector_to_section() {
        let vector = vec![1.0, 2.0, 3.0];
        let entities = create_test_entities();
        let section = SectionMatVecAdapter::dense_vector_to_section(&vector, &entities);

        assert_eq!(section.restrict(&MeshEntity::Face(0)).unwrap().0, 1.0);
        assert_eq!(section.restrict(&MeshEntity::Face(1)).unwrap().0, 2.0);
        assert_eq!(section.restrict(&MeshEntity::Face(2)).unwrap().0, 3.0);
    }

    #[test]
    fn test_section_to_dense_matrix() {
        let section = create_test_section_scalar();
        let matrix = SectionMatVecAdapter::sparse_to_dense_matrix(&section, 3);

        assert_eq!(matrix.read(0, 0), 1.0);
        assert_eq!(matrix.read(1, 1), 2.0);
        assert_eq!(matrix.read(2, 2), 3.0);
    }

    #[test]
    fn test_section_to_matmut() {
        let section = create_test_section_scalar();
        let entity_to_index = create_entity_to_index_map();
        let mat = SectionMatVecAdapter::section_to_matmut(&section, &entity_to_index, 3);

        assert_eq!(mat.read(0, 0), 1.0);
        assert_eq!(mat.read(1, 0), 2.0);
        assert_eq!(mat.read(2, 0), 3.0);
    }

    #[test]
    fn test_matmut_to_section() {
        let mut matrix = Mat::<f64>::zeros(3, 3);
        matrix.write(0, 0, 1.0);
        matrix.write(1, 1, 2.0);
        matrix.write(2, 2, 3.0);
        let entities = create_test_entities();

        let section = SectionMatVecAdapter::matmut_to_section(&matrix, &entities);

        assert_eq!(section.restrict(&MeshEntity::Face(0)).unwrap().0, 1.0);
        assert_eq!(section.restrict(&MeshEntity::Face(1)).unwrap().0, 2.0);
        assert_eq!(section.restrict(&MeshEntity::Face(2)).unwrap().0, 3.0);
    }

    #[test]
    fn test_update_section_with_vector() {
        let mut section = create_test_section_scalar();
        let vector = vec![10.0, 20.0, 30.0];
        let entities = create_test_entities();

        SectionMatVecAdapter::update_section_with_vector(&mut section, &vector, &entities);

        assert_eq!(section.restrict(&MeshEntity::Face(0)).unwrap().0, 10.0);
        assert_eq!(section.restrict(&MeshEntity::Face(1)).unwrap().0, 20.0);
        assert_eq!(section.restrict(&MeshEntity::Face(2)).unwrap().0, 30.0);
    }

    #[test]
    fn test_update_dense_matrix_from_section() {
        let section = create_test_section_scalar();
        let mut matrix = Mat::<f64>::zeros(3, 3);

        SectionMatVecAdapter::update_dense_matrix_from_section(&mut matrix, &section);

        assert_eq!(matrix.read(0, 0), 1.0);
        assert_eq!(matrix.read(1, 1), 2.0);
        assert_eq!(matrix.read(2, 2), 3.0);
    }

    
}

#[cfg(test)]
mod section_to_matmut_tests {
    use super::*;
    use crate::domain::section::{Scalar, Section};
    use crate::domain::mesh_entity::MeshEntity;
    use dashmap::DashMap;

    /// Helper function to create a test section with scalar data.
    fn create_test_section() -> Section<Scalar> {
        let section = Section::new();
        section.set_data(MeshEntity::Face(0), Scalar(1.0));
        section.set_data(MeshEntity::Face(1), Scalar(2.0));
        section.set_data(MeshEntity::Face(2), Scalar(3.0));
        section
    }

    /// Helper function to create an entity-to-index map for testing.
    fn create_test_entity_to_index_map() -> DashMap<MeshEntity, usize> {
        let map = DashMap::new();
        map.insert(MeshEntity::Face(0), 0);
        map.insert(MeshEntity::Face(1), 1);
        map.insert(MeshEntity::Face(2), 2);
        map
    }

    #[test]
    fn test_section_to_matmut_valid_conversion() {
        let section = create_test_section();
        let entity_to_index = create_test_entity_to_index_map();
        let size = 3;

        // Perform the conversion
        let mat = SectionMatVecAdapter::section_to_matmut(&section, &entity_to_index, size);

        // Check the matrix contents
        assert_eq!(mat.read(0, 0), 1.0, "Incorrect value at index 0,0");
        assert_eq!(mat.read(1, 0), 2.0, "Incorrect value at index 1,0");
        assert_eq!(mat.read(2, 0), 3.0, "Incorrect value at index 2,0");
    }

    #[test]
    #[should_panic(expected = "Entity-to-index map contains an index (3) larger than the specified size (3).")]
    fn test_section_to_matmut_index_out_of_bounds() {
        let section = create_test_section();
        let entity_to_index = create_test_entity_to_index_map();

        // Add an invalid index that exceeds the matrix size
        entity_to_index.insert(MeshEntity::Face(3), 3);

        let size = 3;

        // This should panic due to out-of-bounds index
        SectionMatVecAdapter::section_to_matmut(&section, &entity_to_index, size);
    }

    #[test]
    #[should_panic(expected = "Entity Face(3) not found in index map")]
    fn test_section_to_matmut_missing_entity() {
        let section = create_test_section();
        section.set_data(MeshEntity::Face(3), Scalar(4.0)); // Add an entity not in the index map

        let entity_to_index = create_test_entity_to_index_map();
        let size = 3;

        // This should panic due to missing entity in the map
        SectionMatVecAdapter::section_to_matmut(&section, &entity_to_index, size);
    }
}

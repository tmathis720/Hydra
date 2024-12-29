use crate::domain::section::Section;
use crate::domain::mesh_entity::MeshEntity;
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
}

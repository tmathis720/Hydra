// src/interface_adapters/vector_adapter.rs

use crate::linalg::vector::Vector;
use faer::Mat;

/// A struct that adapts vector operations for the Hydra project.
pub struct VectorAdapter;

impl VectorAdapter {
    /// Creates a new dense vector with the specified length.
    pub fn new_dense_vector(size: usize) -> Mat<f64> {
        Mat::zeros(size, 1)
    }

    /// Resizes a vector if the type supports resizing, using specialized handling.
    pub fn resize_vector<T>(vector: &mut T, new_size: usize)
    where
        T: Vector<Scalar = f64> + AsMut<Vec<f64>>,
    {
        vector.as_mut().resize(new_size, 0.0);
    }

    /// Sets an element within the vector.
    pub fn set_element<T: Vector<Scalar = f64>>(vector: &mut T, index: usize, value: f64) {
        vector.set(index, value);
    }

    /// Retrieves an element from the vector.
    pub fn get_element<T: Vector<Scalar = f64>>(vector: &T, index: usize) -> f64 {
        vector.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::vector::vector_builder::VectorBuilder;

    #[test]
    fn test_new_dense_vector() {
        let size = 5;
        let vector = VectorAdapter::new_dense_vector(size);

        assert_eq!(vector.nrows(), size);
        assert_eq!(vector.ncols(), 1);
        for i in 0..size {
            assert_eq!(vector[(i, 0)], 0.0);
        }
    }

    #[test]
    fn test_set_and_get_element() {
        let mut vector = VectorAdapter::new_dense_vector(3);
        VectorAdapter::set_element(&mut vector, 1, 7.0);
        let value = VectorAdapter::get_element(&vector, 1);

        assert_eq!(value, 7.0);
    }

    #[test]
    fn test_resize_vector() {
        let mut vector = VectorBuilder::build_vector::<Vec<f64>>(3);
        VectorAdapter::resize_vector(&mut vector, 5);

        assert_eq!(vector.len(), 5, "Vector length should be updated after resizing.");
    }
}
